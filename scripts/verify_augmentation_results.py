"""Verify a saved augmentation training archive before allowing continuation.

Recomputes validation metrics, verifies source/data/runtime identities and safe
checkpoint state. Does not train, run inference, or read the test partition.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = 'unetpp_rd_aug_001_amp'
sys.path.insert(0, str(ROOT / 'src'))


def verify(archive_path, expected_epochs=None):
    import numpy as np
    import pandas as pd
    import torch
    from build_augmentation_notebook import MODULES
    from rd_ultrasound_ml.unetpp_kaggle import read_official_inputs
    from rd_ultrasound_ml.unetpp_training import (CheckpointStore, binary_metrics,
        file_digest, json_digest, write_json_atomic)

    archive_path = Path(archive_path)
    destination = archive_path.parent / 'verified' / RUN
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        prefix = RUN + '/'
        if (len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or
                '..' in PurePosixPath(n).parts or not n.startswith(prefix) for n in names)):
            raise ValueError('Unsafe or duplicate archive members')
        hashes = json.loads(archive.read(prefix + 'artifact_sha256.json'))
        if set(names) != {prefix + n for n in hashes} | {prefix + 'artifact_sha256.json'}:
            raise ValueError('Archive membership differs from hash manifest')
        for name, expected in hashes.items():
            h = hashlib.sha256()
            with archive.open(prefix + name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError('Artifact hash differs: ' + name)
        for name in [*hashes, 'artifact_sha256.json']:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != archive.read(prefix + name):
                    raise ValueError('Existing verified output differs; retain it for inspection')
            else:
                with archive.open(prefix + name) as source, target.open('wb') as stream:
                    shutil.copyfileobj(source, stream)
    def read(name):
        return json.loads((destination / name).read_text())
    config = json.loads((ROOT / 'configs/unetpp_rd_aug_001_training.json').read_text())
    config['implementation_sha256'] = {n: file_digest(ROOT / f'src/rd_ultrasound_ml/{n}.py') for n in MODULES}
    if config != read('config.json'):
        raise ValueError('Saved training configuration/source identity differs')
    for name, digest in config['implementation_sha256'].items():
        if file_digest(destination / (name + '.py')) != digest:
            raise ValueError('Saved source differs: ' + name)
    runtime, summary, history = read('runtime.json'), read('summary.json'), read('history.json')
    if runtime != config['reference_runtime'] or summary['runtime'] != runtime:
        raise ValueError('Saved runtime differs from the matched baseline')
    epochs = summary['epochs_completed']
    if (not 1 <= epochs <= 50 or (expected_epochs is not None and epochs != expected_epochs) or
            [r['epoch'] for r in history] != list(range(1, epochs + 1)) or
            summary['status'] != ('completed' if epochs == 50 else 'partial') or
            summary['target_epochs'] != 50 or summary['test_used'] is not False):
        raise ValueError('Saved epoch count, status or test policy differs')
    fingerprint = read('input_fingerprint.json')
    record = {k: v for k, v in fingerprint.items() if k not in {'run_fingerprint', 'test_used'}}
    if (fingerprint['preparation_config_digest'] != config['cache']['preparation_config_digest'] or
            fingerprint['cache_fingerprint'] != config['cache']['expected_cache_fingerprint'] or
            fingerprint['training_config_digest'] != json_digest(config) or
            fingerprint['run_fingerprint'] != json_digest(record) or
            summary['data_fingerprint'] != fingerprint['run_fingerprint'] or
            summary['config_digest'] != json_digest(config)):
        raise ValueError('Cached-input or new-run fingerprint differs')
    train, validation = read_official_inputs(ROOT / 'data/erdes')
    for name, frame in [('train', train), ('validation', validation)]:
        saved = pd.read_csv(destination / f'{name}_manifest.csv')
        for key in ('clip_id', 'file_path', 'label'):
            if saved[key].tolist() != frame[key].tolist():
                raise ValueError('Official manifest membership/order/labels differ')
    for row in history:
        prediction = read(f'validation_epoch_{row["epoch"]:03d}.json')
        if prediction['dataset_labels'] != validation['label'].tolist():
            raise ValueError('Validation labels/order differ')
        metrics = binary_metrics(prediction['dataset_labels'], prediction['rd_probabilities'], .5)
        for key, value in metrics.items():
            if isinstance(value, float):
                if not math.isclose(value, row[key], rel_tol=0, abs_tol=1e-12):
                    raise ValueError('Recomputed validation metric differs: ' + key)
            elif row[key] != value:
                raise ValueError('Recomputed validation metric differs: ' + key)
        csv = pd.read_csv(destination / f'validation_epoch_{row["epoch"]:03d}.csv')
        if csv['clip_id'].tolist() != validation['clip_id'].tolist():
            raise ValueError('Prediction CSV clip order differs')
        np.testing.assert_allclose(csv['rd_probability'], prediction['rd_probabilities'], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(csv['dataset_label'], prediction['dataset_labels'])
        np.testing.assert_array_equal(csv['predicted_class'], np.asarray(prediction['rd_probabilities']) >= .5)
    best = max(history, key=lambda r: r['accuracy'])
    if summary['best_epoch'] != best['epoch'] or summary['best_validation_metrics'] != best:
        raise ValueError('Selected checkpoint does not follow the frozen selection rule')
    if read('best_validation_predictions.json') != read(f'validation_epoch_{best["epoch"]:03d}.json'):
        raise ValueError('Selected validation predictions differ')
    checkpoint = CheckpointStore(destination).load()
    if (checkpoint['phase'] != 'epoch_complete' or checkpoint['epoch'] != epochs or
            checkpoint['config'] != config or checkpoint['runtime'] != runtime or
            checkpoint['data_fingerprint'] != summary['data_fingerprint'] or checkpoint['history'] != history):
        raise ValueError('Resume checkpoint identity/progress differs')
    if checkpoint['progress']['position'] != len(train):
        raise ValueError('Resume checkpoint did not finish the epoch')
    best_payload = torch.load(destination / 'best.pt', map_location='cpu', weights_only=True)
    if (best_payload['epoch'] != best['epoch'] or best_payload['config'] != config or
            best_payload['data_fingerprint'] != summary['data_fingerprint']):
        raise ValueError('Selected weight export identity differs')
    for name, tensor in best_payload['model_state'].items():
        if not torch.isfinite(tensor).all() or not torch.equal(tensor, checkpoint['best_model_state'][name]):
            raise ValueError('Selected weights are nonfinite or differ from authoritative checkpoint')
    for tensor in checkpoint['model_state'].values():
        if not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite resume weights')
    updates = epochs * math.ceil(len(train) / config['training']['effective_batch'])
    skipped = sum(r['skipped_updates'] for r in history)
    for state in checkpoint['optimizer_state']['state'].values():
        if int(state['step'].item()) != updates - skipped:
            raise ValueError('Adam step count differs from recorded completed/skipped updates')
        if any(not torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor)):
            raise ValueError('Nonfinite optimizer state')
    reload = read('best_checkpoint_reload.json')
    if not reload['passed'] or reload['checkpoint_sha256'] != file_digest(destination / 'best.pt'):
        raise ValueError('Saved GPU selected-checkpoint reload check failed')
    probe = read('precision_probe.json')
    if probe['status'] != 'passed' or not probe['finite_nonzero_gradients'] or probe['measured_skipped_updates']:
        raise ValueError('GPU precision probe failed')
    report = {'passed': True, 'archive_sha256': file_digest(archive_path), 'verified_artifacts': len(hashes),
        'epochs_completed': epochs, 'selected_epoch': best['epoch'], 'runtime_matches_baseline': True,
        'official_train_rows': len(train), 'validation_rows_per_epoch': len(validation),
        'validation_metrics_recomputed': True, 'resume_checkpoint_verified': True,
        'adam_steps': updates - skipped, 'skipped_updates': skipped,
        'gpu_checkpoint_reload_record_verified': True, 'last_epoch_seconds': history[-1]['seconds_this_segment'],
        'selected_validation_metrics': best, 'verified_directory': str(destination), 'test_used': False,
        'not_run': 'local GPU inference; the complete cache stays on Kaggle'}
    write_json_atomic(archive_path.parent / 'verification.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--expected-epochs', type=int)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.expected_epochs), indent=2))
