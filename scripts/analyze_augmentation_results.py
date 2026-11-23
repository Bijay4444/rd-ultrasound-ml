"""Compare frozen validation predictions; no training, tuning or test access."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def paired_predictions(manifest, baseline, augmented, threshold=.5):
    import numpy as np
    import pandas as pd
    labels = manifest['label'].to_numpy()
    for value in (baseline, augmented):
        p = np.asarray(value['rd_probabilities'])
        if (value['dataset_labels'] != labels.tolist() or p.shape != labels.shape or
                not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()):
            raise ValueError('Prediction labels/order, count or probabilities differ from the manifest')
    result = manifest[['clip_id', 'label']].copy()
    result['baseline_probability'] = baseline['rd_probabilities']
    result['augmentation_probability'] = augmented['rd_probabilities']
    for name in ('baseline', 'augmentation'):
        result[name + '_prediction'] = (result[name + '_probability'] >= threshold).astype(int)
        result[name + '_correct'] = result[name + '_prediction'] == labels
    result['outcome'] = np.select([
        result.baseline_correct & ~result.augmentation_correct,
        ~result.baseline_correct & result.augmentation_correct,
        ~result.baseline_correct & ~result.augmentation_correct],
        ['baseline_correct_augmentation_wrong', 'baseline_wrong_augmentation_correct', 'both_wrong'],
        default='both_correct')
    return result


def read_baseline(archive_path):
    from rd_ultrasound_ml.unetpp_training import binary_metrics, file_digest, json_digest
    from rd_ultrasound_ml.unetpp_kaggle import read_official_inputs
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        prefix = 'unetpp_rd_002_amp/'
        if len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names):
            raise ValueError('Unsafe or duplicate baseline archive members')
        hashes = json.loads(archive.read(prefix + 'artifact_sha256.json'))
        if set(names) != {prefix + n for n in hashes} | {prefix + 'artifact_sha256.json'}:
            raise ValueError('Baseline archive membership differs')
        for name, expected in hashes.items():
            h = hashlib.sha256()
            with archive.open(prefix + name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError('Baseline artifact hash differs: ' + name)
        def read(name):
            return json.loads(archive.read(prefix + name))
        config, summary, history = read('config.json'), read('summary.json'), read('history.json')
        freeze = json.loads((ROOT / 'configs/unetpp_rd_002_evaluation.json').read_text())
        if (summary['status'] != 'completed' or summary['epochs_completed'] != 50 or
                [r['epoch'] for r in history] != list(range(1, 51)) or
                summary['best_epoch'] != max(history, key=lambda r: r['accuracy'])['epoch'] or
                summary['best_epoch'] != freeze['selected_epoch'] or
                hashes['best.pt'] != freeze['checkpoint_sha256'] or
                json_digest(config) != freeze['training_config_digest']):
            raise ValueError('Frozen baseline identity or checkpoint selection differs')
        _, frame = read_official_inputs(ROOT / 'data/erdes')
        for row in history:
            prediction = read(f'validation_epoch_{row["epoch"]:03d}.json')
            if prediction['dataset_labels'] != frame['label'].tolist():
                raise ValueError('Baseline validation labels/order differ')
            metrics = binary_metrics(prediction['dataset_labels'], prediction['rd_probabilities'], .5)
            for key, value in metrics.items():
                if isinstance(value, float):
                    if abs(value - row[key]) > 1e-12:
                        raise ValueError('Recomputed baseline metric differs')
                elif row[key] != value:
                    raise ValueError('Recomputed baseline metric differs')
        prediction = read('best_validation_predictions.json')
        if prediction != read(f'validation_epoch_{summary["best_epoch"]:03d}.json'):
            raise ValueError('Baseline selected predictions differ')
        return config, summary, history, prediction, {'verified_artifacts': len(hashes),
            'archive_sha256': file_digest(archive_path)}


def analyze(augmentation_archive, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_curve, precision_recall_curve
    from rd_ultrasound_ml.unetpp_kaggle import read_official_inputs
    from rd_ultrasound_ml.unetpp_training import write_json_atomic
    from verify_augmentation_results import verify

    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve existing comparison outputs; choose a new directory')
    audit = verify(augmentation_archive, expected_epochs=50)
    augmented_dir = Path(audit['verified_directory'])
    def read_aug(name):
        return json.loads((augmented_dir / name).read_text())
    baseline_archive = ROOT / 'outputs/kaggle_results/session_03/unetpp_rd_002_amp_results.zip'
    base_config, base, base_history, base_prediction, base_audit = read_baseline(baseline_archive)
    augmented_config, augmented, augmented_history = read_aug('config.json'), read_aug('summary.json'), read_aug('history.json')
    for key in ('model', 'seed', 'training', 'preprocessing', 'dataset_revision', 'task', 'label_mapping'):
        if base_config[key] != augmented_config[key]:
            raise ValueError('Matched augmentation control differs: ' + key)
    if base['runtime'] != augmented['runtime']:
        raise ValueError('Control and augmented runtime differ')
    _, validation = read_official_inputs(ROOT / 'data/erdes')
    aug_prediction = read_aug('best_validation_predictions.json')
    paired = paired_predictions(validation, base_prediction, aug_prediction)
    output.mkdir(parents=True)
    paired.to_csv(output / 'paired_validation_predictions.csv', index=False)
    paired[paired.outcome != 'both_correct'].to_csv(output / 'validation_error_cases.csv', index=False)
    rate_keys = ['sensitivity', 'specificity', 'precision', 'F1', 'accuracy', 'balanced_accuracy', 'average_precision', 'ROC_AUC']
    metrics = []
    b, a = base['best_validation_metrics'], augmented['best_validation_metrics']
    for key in rate_keys + ['TP', 'TN', 'FP', 'FN']:
        metrics.append({'metric': key, 'baseline': b[key], 'augmentation': a[key],
            'difference': a[key] - b[key], 'difference_percentage_points': 100 * (a[key] - b[key]) if key in rate_keys else None})
    pd.DataFrame(metrics).to_csv(output / 'validation_metrics.csv', index=False)
    colors = ['#155e75', '#b45309']
    labels = [f'Baseline · selected epoch {base["best_epoch"]}', f'Augmentation · selected epoch {augmented["best_epoch"]}']
    def save(name, figure):
        figure.tight_layout(rect=(0, 0, 1, .95))
        figure.savefig(output / (name + '.png'), dpi=180, bbox_inches='tight')
        figure.savefig(output / (name + '.pdf'), bbox_inches='tight')
        plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for history, name, color, selected in zip([base_history, augmented_history], labels, colors, [base['best_epoch'], augmented['best_epoch']]):
        for axis, field, title in zip(axes.flat, ['train_loss', 'validation_loss', 'sensitivity', 'average_precision'],
                ['Training BCE (augmented inputs in new run)', 'Validation BCE (clean inputs)', 'Validation RD sensitivity', 'Validation average precision']):
            axis.plot([r['epoch'] for r in history], [r[field] for r in history], label=name, color=color)
            axis.axvline(selected, color=color, linestyle=':', alpha=.6)
            axis.set(xlabel='Epoch', title=title)
            axis.grid(alpha=.18)
    for axis in axes.flat:
        axis.legend(fontsize=8)
    figure.suptitle('Same 50-epoch budget · seed 42 · fixed validation split and selection rule')
    save('learning_curves', figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, metric, title in zip(axes, [b, a], labels):
        counts = np.asarray([[metric['TN'], metric['FP']], [metric['FN'], metric['TP']]])
        normalized = counts / counts.sum(axis=1, keepdims=True)
        axis.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f'{counts[row, column]}\n{normalized[row, column]:.1%}',
                    ha='center', va='center', fontsize=14, color='white' if normalized[row, column] > .5 else '#18313b')
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['non-RD', 'RD'], yticklabels=['non-RD', 'RD'],
            xlabel='Predicted label', ylabel='Dataset label', title=title)
    figure.suptitle('Validation · threshold 0.5 · counts and within-label percentages')
    save('confusion_matrices', figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    y = validation['label'].to_numpy()
    for prediction, metric, name, color in zip([base_prediction, aug_prediction], [b, a], labels, colors):
        fpr, tpr, _ = roc_curve(y, prediction['rd_probabilities'])
        precision, recall, _ = precision_recall_curve(y, prediction['rd_probabilities'])
        axes[0].plot(fpr, tpr, color=color, label=f'{name}; AUC {metric["ROC_AUC"]:.4f}')
        axes[1].step(recall, precision, where='post', color=color, label=f'{name}; AP {metric["average_precision"]:.4f}')
    axes[0].plot([0, 1], [0, 1], linestyle=':', color='#64748b')
    axes[1].axhline(y.mean(), linestyle=':', color='#64748b', label='Validation RD prevalence')
    axes[0].set(xlabel='False-positive rate', ylabel='RD sensitivity', title='Validation ROC', xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel='RD recall', ylabel='Precision', title='Validation precision–recall (AP)', xlim=(0, 1), ylim=(0, 1.02))
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.18)
    save('roc_precision_recall', figure)
    report = {'status': 'completed_validation_comparison', 'training_epochs': 50,
        'baseline_selected_epoch': base['best_epoch'], 'augmentation_selected_epoch': augmented['best_epoch'],
        'threshold': .5, 'seed': 42, 'validation_clips': len(validation), 'metrics': metrics,
        'paired_outcomes': {str(k): int(v) for k, v in paired.outcome.value_counts().items()},
        'baseline_epoch_seconds_sum': sum(r['seconds_this_segment'] for r in base_history),
        'augmentation_epoch_seconds_sum': sum(r['seconds_this_segment'] for r in augmented_history),
        'timing_definition': 'sum of training+validation epoch segments; excludes setup, probes, restart/export overhead',
        'baseline_skipped_updates': sum(r['skipped_updates'] for r in base_history),
        'augmentation_skipped_updates': sum(r['skipped_updates'] for r in augmented_history),
        'baseline_audit': base_audit, 'augmentation_audit': audit, 'test_used': False,
        'interpretation': 'The selected augmented run did not improve sensitivity, specificity or AP on this validation split. One seed does not establish a general effect.'}
    write_json_atomic(output / 'comparison.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.archive, args.output)
    print(json.dumps({k: result[k] for k in ['status', 'baseline_selected_epoch', 'augmentation_selected_epoch', 'paired_outcomes', 'interpretation']}, indent=2))
