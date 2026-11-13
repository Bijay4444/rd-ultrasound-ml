"""Frozen-checkpoint evaluation of comparison models on the reused official test split.

Model-agnostic (R(2+1)D-18, R3D-18). It reuses the completed U-Net++ test-input
cache without relabeling its freeze: the cache's own freeze must hash to its
manifest digest and to a pinned digest, the official split archive is re-checked,
and every tensor checksum, label, order and preprocessing identity is verified.
The training artifact is verified before the test CSV is opened. No optimizer,
threshold search, epoch selection, calibration or retraining exists here.
"""
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rd_ultrasound_ml.unetpp_data import CachedClips
from rd_ultrasound_ml.unetpp_evaluation import validate_test_partition
from rd_ultrasound_ml.unetpp_kaggle import read_official_inputs
from rd_ultrasound_ml.unetpp_training import (
    autocast_for, binary_metrics, file_digest, json_digest, write_json_atomic,
)

BENCHMARK_LABEL = 'reused official test benchmark'


def verify_comparison_training(directory, freeze):
    """Check the team freeze and the selected training artifact; must precede any test access."""
    directory = Path(directory)
    if freeze.get('team_freeze_confirmed') is not True:
        raise ValueError('Team has not confirmed the frozen model choices; set team_freeze_confirmed '
                         'in the evaluation config in a reviewed commit before any test access')
    if (freeze['training_epochs_completed'] != 50 or freeze['threshold'] != .5
            or freeze['test_accessed_at_freeze'] is not False or freeze['checkpoint_file'] != 'best.pt'):
        raise ValueError('Invalid pre-test freeze: require complete training and the fixed protocol')
    config = json.loads((directory/'config.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    if (config.get('experiment_id') != freeze['training_experiment_id']
            or config.get('implementation_sha256') != freeze['training_implementation_sha256']):
        raise ValueError('Training experiment or implementation hashes differ from freeze')
    if (json_digest(config) != freeze['training_config_digest']
            or summary['config_digest'] != freeze['training_config_digest']
            or summary['data_fingerprint'] != freeze['training_run_fingerprint']):
        raise ValueError('Frozen training configuration/run fingerprint differs')
    if (summary['status'] != 'completed' or summary['epochs_completed'] != 50
            or summary['best_epoch'] != freeze['selected_epoch'] or summary['test_used']):
        raise ValueError('Training is incomplete or selected epoch differs from freeze')
    identity = json.loads((directory/'input_fingerprint.json').read_text())['preprocessing_identity']
    if identity != freeze['preprocessing_identity']:
        raise ValueError('Training preprocessing identity differs from freeze')
    if file_digest(directory/freeze['checkpoint_file']) != freeze['checkpoint_sha256']:
        raise ValueError('Selected checkpoint SHA256 differs from pre-test freeze')
    return config


def find_comparison_training(input_root, freeze):
    matches = [path.parent for path in Path(input_root).rglob('best.pt')
               if file_digest(path) == freeze['checkpoint_sha256']]
    if len(matches) != 1:
        raise ValueError('Attach exactly one matching final-training output for the frozen checkpoint')
    verify_comparison_training(matches[0], freeze)
    return matches[0]


def load_reused_test_cache(input_root, training_directory, freeze, report=print):
    """Verify the reused test-input cache under this model's freeze; return frame, dataset, record."""
    verify_comparison_training(training_directory, freeze)  # Must precede opening test.csv.
    manifests = list(Path(input_root).rglob('test_prepared_manifest.json'))
    if len(manifests) != 1:
        raise ValueError('Attach exactly one successfully saved test-preparation output')
    root = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    preparation_freeze = json.loads((root/'evaluation_freeze.json').read_text())
    pinned = freeze['test_cache']['preparation_freeze_digest']
    if (manifest.get('complete') is not True or json_digest(preparation_freeze) != manifest['freeze_digest']
            or manifest['freeze_digest'] != pinned):
        raise ValueError('Test cache is incomplete or its preparation freeze differs from the pin')
    train, validation = read_official_inputs(root)  # Checks official archive/metadata MD5s.
    with zipfile.ZipFile(root/'splits.zip') as archive:
        content = archive.read('splits/non_rd_vs_rd/test.csv')
    split_hash = hashlib.sha256(content).hexdigest()
    metadata = pd.read_csv(root/'erdes_metadata.csv')
    frame = validate_test_partition(pd.read_csv(io.BytesIO(content)), metadata, train, validation)
    if manifest['test_csv_sha256'] != split_hash or len(manifest['entries']) != len(frame):
        raise ValueError('Test cache differs from the official test split')
    if len(frame) != freeze['test_clips'] or frame['label'].value_counts().to_dict() != {
            int(key): value for key, value in freeze['test_label_counts'].items()}:
        raise ValueError('Official test count/labels differ from freeze')
    identity = manifest['preprocessing_identity']
    if identity != freeze['preprocessing_identity']:
        raise ValueError('Test preprocessing identity differs from the training inputs')
    signature = json_digest(identity)
    paths = []
    for index, row in enumerate(frame.itertuples(index=False)):
        entry = manifest['entries'][index]
        expected = {'signature': signature, 'clip_id': row.clip_id, 'file_path': row.file_path,
            'source_sha256': entry['source']['sha256'], 'frame_count': int(row.frame_count),
            'height': int(row.height), 'width': int(row.width), 'target': [96, 128, 128]}
        if (entry['index'] != index or entry['clip_id'] != row.clip_id or entry['label'] != int(row.label)
                or entry['check']['identity'] != expected):
            raise ValueError('Test cache identity/order/source provenance differs')
        relative = Path(entry['array'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe test cache path')
        path = root/relative
        if file_digest(path) != entry['check']['tensor_sha256']:
            raise ValueError(f'Test tensor checksum differs: {row.clip_id}')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (1, 96, 128, 128):
            raise ValueError('Test tensor shape/dtype differs')
        paths.append(path)
        if (index+1) % 100 == 0 or index+1 == len(frame):
            report(f'Verified test input {index+1}/{len(frame)}')
    record = {'freeze_digest': json_digest(freeze), 'preparation_freeze_digest': pinned,
        'test_csv_sha256': split_hash, 'prepared_manifest_sha256': json_digest(manifest),
        'preprocessing_identity': identity, 'test_clips': len(frame)}
    return frame, CachedClips(paths, frame['label']), record


def evaluate_comparison_model(model_factory, dataset, frame, training_directory, freeze,
                              output_directory, runtime, device='cuda', report=print):
    """One forward pass per test clip at the frozen threshold; records inference timing."""
    verify_comparison_training(training_directory, freeze)
    if len(dataset) != len(frame) or len(frame) != freeze['test_clips']:
        raise ValueError('Test dataset/manifest count differs from freeze')
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    if (output_directory/'test_metrics.json').exists():
        raise ValueError('Test results already exist; inspect them rather than rerunning')
    payload = torch.load(Path(training_directory)/freeze['checkpoint_file'], map_location='cpu', weights_only=True)
    if (payload['epoch'] != freeze['selected_epoch'] or json_digest(payload['config']) != freeze['training_config_digest']
            or payload['data_fingerprint'] != freeze['training_run_fingerprint'] or payload.get('test_used')):
        raise ValueError('Checkpoint payload differs from frozen selection')
    device = torch.device(device)
    model = model_factory().to(device)
    model.load_state_dict(payload['model_state'])
    model.eval()
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    loader = DataLoader(dataset, batch_size=freeze['physical_batch'], shuffle=False, num_workers=0,
                        pin_memory=device.type == 'cuda')
    probabilities, labels, forward_seconds, loss_sum = [], [], [], 0.
    criterion = torch.nn.BCEWithLogitsLoss()
    synchronize = torch.cuda.synchronize if device.type == 'cuda' else (lambda: None)
    start = time.perf_counter()
    with torch.inference_mode():
        for inputs, target in loader:
            inputs, target = inputs.to(device), target.to(device)
            synchronize()
            began = time.perf_counter()
            with autocast_for(device, freeze['precision']):
                logits = model(inputs).reshape(-1)
            synchronize()
            forward_seconds.append(time.perf_counter()-began)
            loss = criterion(logits.float(), target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite test inference loss')
            probabilities.extend(torch.sigmoid(logits.float()).cpu().tolist())
            labels.extend(target.cpu().tolist())
            loss_sum += loss.item()*len(target)
            if len(labels) % 100 == 0 or len(labels) == len(frame):
                report(f'Inferred test clips {len(labels)}/{len(frame)}')
    total_seconds = time.perf_counter()-start
    if not np.array_equal(np.asarray(labels), frame['label'].to_numpy()):
        raise ValueError('Inference label order differs from official test manifest')
    if any(not torch.equal(value.detach().cpu(), before[name]) for name, value in model.state_dict().items()):
        raise ValueError('Model state changed during test evaluation')
    metrics = binary_metrics(np.asarray(labels), np.asarray(probabilities), freeze['threshold'])
    prediction = frame[['clip_id', 'file_path', 'label']].copy()
    prediction['rd_probability'] = probabilities
    prediction['predicted_label'] = (prediction['rd_probability'] >= freeze['threshold']).astype(int)
    prediction.to_csv(output_directory/'test_predictions.csv', index=False)
    frame.to_csv(output_directory/'test_manifest.csv', index=False)
    first_clips = min(freeze['physical_batch'], len(frame))
    steady_seconds, steady_clips = ((sum(forward_seconds[1:]), len(frame)-first_clips) if len(forward_seconds) > 1
                                    else (sum(forward_seconds), len(frame)))
    timing = {'hardware': runtime.get('gpu') or str(device), 'physical_batch': freeze['physical_batch'],
        'precision': freeze['precision'], 'clips': len(frame),
        'seconds_including_disk_loading': total_seconds,
        'model_forward_seconds': sum(forward_seconds), 'first_batch_seconds': forward_seconds[0],
        'model_forward_ms_per_clip_excluding_first_batch': 1000*steady_seconds/steady_clips}
    result = {'evaluation_id': freeze['evaluation_id'], 'model': freeze['model_label'],
        'partition': 'official_test', 'benchmark_label': BENCHMARK_LABEL, 'test_clips': len(frame),
        'selected_epoch': freeze['selected_epoch'], 'checkpoint_sha256': freeze['checkpoint_sha256'],
        'freeze_digest': json_digest(freeze), 'precision': freeze['precision'], 'runtime': runtime,
        'loss': loss_sum/len(frame), 'inference_timing': timing, 'metrics': metrics,
        'test_used': True, 'training_performed': False, 'threshold_tuned': False}
    write_json_atomic(output_directory/'test_metrics.json', result)
    write_json_atomic(output_directory/'evaluation_freeze.json', freeze)
    rows = [{'source': freeze['model_label'] + ' (this evaluation)', 'partition': BENCHMARK_LABEL, **metrics}]
    rows += [dict(row) for row in freeze.get('reference_test_rows', [])]
    pd.DataFrame(rows).to_csv(output_directory/'test_reference_comparison.csv', index=False)
    return result
