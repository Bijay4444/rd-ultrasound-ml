"""Sample-weighted accumulation and resumable UNet++ training infrastructure.

Own implementation; the ERDES reference supplies the task/model/protocol choices.
Checkpoints contain plain containers and tensors and load with weights_only=True.
"""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score


def json_digest(value):
    content = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_random_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': {
        'algorithm': state[0], 'keys': torch.from_numpy(state[1].astype(np.int64)),
        'position': state[2], 'has_gaussian': state[3], 'cached_gaussian': state[4]},
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(value):
    random.setstate(value['python'])
    state = value['numpy']
    np.random.set_state((state['algorithm'], state['keys'].cpu().numpy().astype(np.uint32),
                         state['position'], state['has_gaussian'], state['cached_gaussian']))
    torch.set_rng_state(value['torch'].cpu())
    if value['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in value['cuda']])


def epoch_order(length, seed, epoch):
    if length < 1 or epoch < 1:
        raise ValueError('Dataset length and epoch must be positive')
    generator = torch.Generator().manual_seed(seed + epoch - 1)
    return torch.randperm(length, generator=generator).tolist()


def autocast_for(device, precision):
    device = torch.device(device)
    if precision not in {'float32', '16-mixed'}:
        raise ValueError('Expected float32 or 16-mixed precision')
    if precision == '16-mixed':
        if device.type != 'cuda':
            raise ValueError('16-mixed training requires a CUDA device')
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()


def train_epoch_accumulated(model, dataset, order, optimizer, device, physical_batch,
                            effective_batch, precision='float32', scaler=None,
                            start_position=0, loss_sum=0., examples_seen=0,
                            optimizer_updates=0, skipped_updates=0, on_boundary=None):
    """Resume only at clean optimizer boundaries; normalize the short final window."""
    device = torch.device(device)
    length = len(dataset)
    if sorted(order) != list(range(length)):
        raise ValueError('Epoch order must contain every dataset index exactly once')
    if physical_batch < 1 or effective_batch < 1 or effective_batch % physical_batch:
        raise ValueError('Effective batch must be a positive multiple of physical batch')
    if not (0 <= start_position <= length) or (start_position % effective_batch and start_position != length):
        raise ValueError('Resume position must be an optimizer-update boundary')
    if examples_seen != start_position:
        raise ValueError('Saved epoch example count differs from its cursor')
    if precision == '16-mixed' and (scaler is None or not scaler.is_enabled()):
        raise ValueError('Mixed precision requires an enabled GradScaler')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # An explicit generator avoids a new iterator consuming the model's global RNG on resume.
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(dataset, batch_size=physical_batch, sampler=order[start_position:],
                        num_workers=0, pin_memory=device.type == 'cuda', generator=generator)
    position = start_position
    window_start = start_position
    loss_function = nn.BCEWithLogitsLoss()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        window_size = min(effective_batch, length - window_start)
        with autocast_for(device, precision):
            logits = model(inputs).reshape(-1)
            loss = loss_function(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss; resume from the last valid checkpoint')
        normalized = loss * (len(labels) / window_size)
        if precision == '16-mixed':
            scaler.scale(normalized).backward()
        else:
            normalized.backward()
        loss_sum += loss.item() * len(labels)
        position += len(labels)
        examples_seen += len(labels)
        if position == window_start + window_size:
            if precision == '16-mixed':
                # Keep one scale throughout the entire effective batch. Step/update only here.
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped_updates += int(scaler.get_scale() < old_scale)
            else:
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite FP32 gradients; optimizer was not stepped')
                optimizer.step()
            optimizer_updates += 1
            optimizer.zero_grad(set_to_none=True)
            progress = {'position': position, 'loss_sum': loss_sum, 'examples_seen': examples_seen,
                        'optimizer_updates': optimizer_updates, 'skipped_updates': skipped_updates}
            if on_boundary is not None:
                on_boundary(progress.copy())
            window_start = position
    return {'position': position, 'loss_sum': loss_sum, 'examples_seen': examples_seen,
            'optimizer_updates': optimizer_updates, 'skipped_updates': skipped_updates,
            'train_loss': loss_sum / length}


def validate_model(model, dataset, device, physical_batch, precision='float32'):
    device = torch.device(device)
    model.eval()
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(dataset, batch_size=physical_batch, shuffle=False, num_workers=0,
                        pin_memory=device.type == 'cuda', generator=generator)
    loss_sum = 0.
    labels, probabilities = [], []
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for inputs, target in loader:
            inputs, target = inputs.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with autocast_for(device, precision):
                logits = model(inputs).reshape(-1)
                loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite validation loss')
            loss_sum += loss.item() * len(target)
            labels.extend(target.cpu().tolist())
            probabilities.extend(torch.sigmoid(logits.float()).cpu().tolist())
    return loss_sum / len(dataset), np.asarray(labels), np.asarray(probabilities)


def binary_metrics(labels, probabilities, threshold=.5):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or len(labels) == 0 or labels.shape != probabilities.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected non-empty matching binary labels and probability vectors')
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Probabilities must be finite and in [0,1]')
    prediction = probabilities >= threshold
    tp, tn = int(np.sum((labels == 1) & prediction)), int(np.sum((labels == 0) & ~prediction))
    fp, fn = int(np.sum((labels == 0) & prediction)), int(np.sum((labels == 1) & ~prediction))
    reasons = {}
    def divide(name, numerator, denominator):
        if denominator == 0:
            reasons[name] = 'zero denominator'
            return None
        return numerator / denominator
    sensitivity = divide('sensitivity', tp, tp + fn)
    specificity = divide('specificity', tn, tn + fp)
    value = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'sensitivity': sensitivity,
        'specificity': specificity, 'precision': divide('precision', tp, tp + fp),
        'F1': divide('F1', 2 * tp, 2 * tp + fp + fn), 'accuracy': (tp + tn) / len(labels),
        'balanced_accuracy': None if sensitivity is None or specificity is None else (sensitivity + specificity) / 2,
        'average_precision': None, 'ROC_AUC': None, 'threshold': threshold, 'positive_class': 'rd'}
    if set(labels.tolist()) == {0, 1}:
        value['average_precision'] = float(average_precision_score(labels, probabilities))
        value['ROC_AUC'] = float(roc_auc_score(labels, probabilities))
    else:
        reasons.update(average_precision='requires both classes for this protocol', ROC_AUC='requires both classes',
                       balanced_accuracy='requires both classes')
    value['undefined_reasons'] = reasons
    return value


def probe_training_precision(model_factory, inputs, labels, physical_batch=4, precision='16-mixed'):
    """Disposable real-input timing; stabilize the scaler before measured updates."""
    if not torch.cuda.is_available():
        raise RuntimeError('The precision probe requires CUDA')
    model = optimizer = x = y = loss = scaler = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = model_factory().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-5)
        scaler = torch.amp.GradScaler('cuda', enabled=precision == '16-mixed')
        indices = torch.arange(physical_batch) % len(inputs)
        x, y = inputs[indices].cuda(), labels[indices].cuda()
        timings, skipped_measured = [], 0
        for iteration in range(13):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with autocast_for('cuda', precision):
                loss = nn.BCEWithLogitsLoss()(model(x).reshape(-1), y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite precision-probe loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            nonzero = finite and any(torch.count_nonzero(p.grad).item() for p in model.parameters())
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            if iteration >= 8:
                skipped_measured += int(scaler.get_scale() < old_scale)
                if not finite or not nonzero:
                    raise ValueError('Precision probe did not stabilize with finite nonzero gradients')
                timings.append(time.perf_counter() - start)
        if skipped_measured:
            raise ValueError('Measured precision-probe updates were skipped')
        return {'status': 'passed', 'precision': precision, 'physical_batch': physical_batch,
            'training_step_seconds_median': float(np.median(timings)),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'loss_last': loss.item(), 'final_scale': scaler.get_scale(),
            'measured_skipped_updates': skipped_measured, 'finite_nonzero_gradients': True,
            'weights_discarded': True, 'test_used': False}
    finally:
        del model, optimizer, x, y, loss, scaler
        torch.cuda.empty_cache()


class CheckpointStore:
    """Two alternating files plus an atomic pointer retain the last valid generation."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pointer = self.directory / 'last_checkpoint.json'

    def save(self, payload):
        previous = json.loads(self.pointer.read_text()) if self.pointer.exists() else None
        slot = 'last_b.pt' if previous and previous['file'] == 'last_a.pt' else 'last_a.pt'
        target = self.directory / slot
        temporary = target.with_suffix('.pt.tmp')
        torch.save(payload, temporary)
        digest = file_digest(temporary)
        temporary.replace(target)
        write_json_atomic(self.pointer, {'file': slot, 'sha256': digest,
            'previous': None if previous is None else {'file': previous['file'], 'sha256': previous['sha256']}})

    def load(self):
        if not self.pointer.exists():
            return None
        current = json.loads(self.pointer.read_text())
        for candidate in [current, current.get('previous')]:
            if not candidate:
                continue
            path = self.directory / candidate['file']
            if candidate['file'] not in {'last_a.pt', 'last_b.pt'}:
                raise ValueError('Unexpected checkpoint filename')
            if path.is_file() and file_digest(path) == candidate['sha256']:
                payload = torch.load(path, map_location='cpu', weights_only=True)
                if candidate is not current:
                    # Repair the pointer before any save can overwrite the fallback generation.
                    write_json_atomic(self.pointer, {**candidate, 'previous': None})
                return payload
        raise ValueError('No valid checkpoint generation; files and hashes disagree')


def run_training(model_factory, train_dataset, validation_dataset, config, directory,
                 data_fingerprint, runtime, device='cuda', max_epochs_this_call=1, report=print):
    """Resume a fixed protocol, including interrupted epochs at update boundaries."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    signature = json_digest(config)
    store = CheckpointStore(directory)
    checkpoint = store.load()
    if checkpoint is not None:
        if checkpoint['schema_version'] != 1 or checkpoint['config_digest'] != signature:
            raise ValueError('Checkpoint configuration differs; choose a new run directory')
        if checkpoint['data_fingerprint'] != data_fingerprint:
            raise ValueError('Checkpoint data/decoder/cache fingerprint differs; do not resume on changed inputs')
        if checkpoint['runtime'] != runtime:
            raise ValueError('Runtime versions/GPU differ from the checkpoint; inspect before changing the fixed run')
    elif (directory / 'history.json').exists():
        raise ValueError('History exists without a valid checkpoint; retain artifacts and inspect before restarting')
    device = torch.device(device)
    precision = config['training']['precision']
    if precision == '16-mixed' and device.type != 'cuda':
        raise ValueError('Mixed-precision training must use CUDA')
    seed_training(config['seed'])
    model = model_factory().to(device)
    settings = config['training']
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'],
        betas=tuple(settings['betas']), eps=settings['eps'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        factor=.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0., eps=1e-8)
    scaler = torch.amp.GradScaler('cuda', enabled=precision == '16-mixed')
    history, best_accuracy, best_epoch, best_state = [], None, None, None
    first_epoch, resume_progress = 1, {}
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        history, best_accuracy, best_epoch = checkpoint['history'], checkpoint['best_accuracy'], checkpoint['best_epoch']
        best_state = checkpoint['best_model_state']
        first_epoch = checkpoint['epoch'] + int(checkpoint['phase'] == 'epoch_complete')
        if checkpoint['phase'] == 'train':
            resume_progress = checkpoint['progress']
        restore_random_state(checkpoint['random_state'])
        report(f'Resuming epoch {first_epoch}, training cursor {resume_progress.get("position", 0)}')
    # Rebuild human-readable history from the authoritative checkpoint after interrupted writes.
    write_json_atomic(directory / 'history.json', history)
    write_json_atomic(directory / 'resolved_optimizer.json', {'optimizer': optimizer.__class__.__name__,
        'defaults': optimizer.defaults, 'scheduler': {'name': 'ReduceLROnPlateau',
        **{key: value for key, value in scheduler.state_dict().items()
           if key in {'mode', 'factor', 'patience', 'threshold', 'threshold_mode', 'cooldown', 'min_lrs', 'eps'}}},
        'precision': precision})
    target_epochs = settings['epochs']
    stop_epoch = target_epochs if max_epochs_this_call is None else min(target_epochs, first_epoch + max_epochs_this_call - 1)
    if max_epochs_this_call is not None and max_epochs_this_call < 1:
        raise ValueError('Session epoch limit must be positive or None')
    training_start = time.perf_counter()
    for epoch in range(first_epoch, stop_epoch + 1):
        epoch_start = time.perf_counter()
        resumed_from = resume_progress.get('position', 0)
        order = epoch_order(len(train_dataset), config['seed'], epoch)
        if checkpoint is not None and resume_progress and checkpoint['order'] != order:
            raise ValueError('Saved epoch order differs from the fixed seed/order protocol')
        def make_payload(phase, progress):
            return {'schema_version': 1, 'config_digest': signature, 'config': config,
                'data_fingerprint': data_fingerprint, 'runtime': runtime,
                'epoch': epoch, 'phase': phase, 'order': order, 'progress': progress,
                'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(), 'scaler_state': scaler.state_dict(),
                'history': history, 'best_accuracy': best_accuracy, 'best_epoch': best_epoch,
                'best_model_state': best_state, 'random_state': capture_random_state(), 'test_used': False}
        # A checkpoint exists before the first batch, including the initialized model.
        if not resume_progress:
            resume_progress = {'position': 0, 'loss_sum': 0., 'examples_seen': 0,
                               'optimizer_updates': 0, 'skipped_updates': 0}
            store.save(make_payload('train', resume_progress))
        def boundary(progress):
            if progress['optimizer_updates'] % settings['checkpoint_every_updates'] == 0 or progress['position'] == len(train_dataset):
                store.save(make_payload('train', progress))
                write_json_atomic(directory / 'run_status.json', {'status': 'training',
                    'epoch': epoch, 'training_examples_seen': progress['position'],
                    'training_examples_total': len(train_dataset), 'completed_epochs': len(history), 'test_used': False})
                report(f'Epoch {epoch}/{target_epochs}: {progress["position"]}/{len(train_dataset)} training clips; checkpoint saved')
            elif progress['optimizer_updates'] % 2 == 0:
                report(f'Epoch {epoch}/{target_epochs}: {progress["position"]}/{len(train_dataset)} training clips')
        progress = train_epoch_accumulated(model, train_dataset, order, optimizer, device,
            settings['physical_batch'], settings['effective_batch'], precision, scaler,
            on_boundary=boundary, **{key: value for key, value in resume_progress.items() if key != 'position'},
            start_position=resume_progress['position'])
        validation_start = time.perf_counter()
        validation_loss, labels, probabilities = validate_model(model, validation_dataset, device,
            settings['physical_batch'], precision)
        metrics = binary_metrics(labels, probabilities, config['evaluation']['threshold'])
        row = {'epoch': epoch, 'train_loss': progress['train_loss'], 'validation_loss': validation_loss,
            'resumed_from_position': resumed_from,
            'learning_rate_used': optimizer.param_groups[0]['lr'], 'seconds_this_segment': time.perf_counter() - epoch_start,
            'validation_seconds': time.perf_counter() - validation_start,
            'skipped_updates': progress['skipped_updates'], **metrics}
        scheduler.step(validation_loss)
        history.append(row)
        is_best = best_accuracy is None or metrics['accuracy'] > best_accuracy
        if is_best:
            best_accuracy, best_epoch = metrics['accuracy'], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        prediction = {'dataset_labels': labels.tolist(), 'rd_probabilities': probabilities.tolist()}
        write_json_atomic(directory / f'validation_epoch_{epoch:03d}.json', prediction)
        store.save(make_payload('epoch_complete', progress))
        write_json_atomic(directory / 'history.json', history)
        if is_best:
            write_json_atomic(directory / 'best_validation_predictions.json', prediction)
        report({key: row[key] for key in ['epoch', 'train_loss', 'validation_loss', 'accuracy',
                                        'sensitivity', 'specificity', 'average_precision', 'seconds_this_segment']})
        resume_progress = {}
    # best.pt is a derived export; authoritative best weights also live in both latest checkpoint generations.
    if best_state is not None:
        temporary = directory / 'best.pt.tmp'
        torch.save({'model_state': best_state, 'epoch': best_epoch, 'config': config,
                    'data_fingerprint': data_fingerprint, 'test_used': False}, temporary)
        temporary.replace(directory / 'best.pt')
        best_prediction = directory / f'validation_epoch_{best_epoch:03d}.json'
        write_json_atomic(directory / 'best_validation_predictions.json', json.loads(best_prediction.read_text()))
    completed = len(history)
    summary = {'experiment_id': config['experiment_id'], 'epochs_completed': completed,
        'target_epochs': target_epochs, 'status': 'completed' if completed == target_epochs else 'partial',
        'best_epoch': best_epoch, 'best_validation_accuracy': best_accuracy,
        'best_validation_metrics': next((row for row in history if row['epoch'] == best_epoch), None),
        'config_digest': signature, 'data_fingerprint': data_fingerprint, 'runtime': runtime,
        'precision': precision, 'training_clips': len(train_dataset), 'validation_clips': len(validation_dataset),
        'last_call_seconds': time.perf_counter() - training_start, 'test_used': False}
    write_json_atomic(directory / 'summary.json', summary)
    write_json_atomic(directory / 'run_status.json', summary)
    return summary
