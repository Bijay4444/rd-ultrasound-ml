"""Verify saved test archives and plot the four frozen experiments; no inference.

Requires the previously downloaded, version-pinned outputs documented in the
result reports. Patient independence and repeat-seed uncertainty are not tested.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read_archive(path):
    with zipfile.ZipFile(path) as archive:
        entries = [p for p in archive.infolist() if not p.is_dir()]
        names = [p.filename for p in entries]
        if (len(names) != len(set(names)) or sum(p.file_size for p in entries) > 16_000_000
                or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names)
                or any(stat.S_ISLNK(p.external_attr >> 16) for p in entries)):
            raise ValueError('Unsafe test archive')
        manifests = [n for n in names if n.endswith('/artifact_sha256.json')]
        if len(manifests) != 1:
            raise ValueError('Expected one artifact manifest')
        prefix = manifests[0].rsplit('/', 1)[0] + '/'
        if not all(n.startswith(prefix) for n in names):
            raise ValueError('Multiple archive roots')
        files = {n[len(prefix):]: archive.read(n) for n in names}
    hashes = json.loads(files['artifact_sha256.json'])
    if set(files) != {*hashes, 'artifact_sha256.json'}:
        raise ValueError('Unlisted test archive members')
    for name, expected in hashes.items():
        if hashlib.sha256(files[name]).hexdigest() != expected:
            raise ValueError('Test artifact checksum mismatch: ' + name)
    return files, {'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                   'artifact_hashes_verified': len(hashes)}


def metrics(frame):
    y, p = frame.label.to_numpy(), frame.rd_probability.to_numpy()
    if (set(y) != {0, 1} or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
            or not np.array_equal((p >= .5).astype(int), frame.predicted_label.to_numpy())):
        raise ValueError('Invalid saved test probabilities or threshold decisions')
    tn, fp, fn, tp = map(int, confusion_matrix(y, p >= .5, labels=[0, 1]).ravel())
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, sensitivity=tp/(tp+fn),
        specificity=tn/(tn+fp), precision=tp/(tp+fp) if tp+fp else None,
        F1=2*tp/(2*tp+fp+fn), accuracy=(tp+tn)/len(y),
        balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,
        average_precision=average_precision_score(y, p), ROC_AUC=roc_auc_score(y, p))


def compare(augmentation_archive, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    def read(path):
        return json.loads(Path(path).read_text())
    for filename, expected in [('splits.zip', 'c5630e3f056ce358dcd384aaed946e64'),
                               ('erdes_metadata.csv', '95828fb5bdffdbf8173b5bcfc99e8148')]:
        if hashlib.md5((ROOT/'data/erdes'/filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Original release checksum differs: ' + filename)
    with zipfile.ZipFile(ROOT / 'data/erdes/splits.zip') as archive:
        official_content = archive.read('splits/non_rd_vs_rd/test.csv')
    official = pd.read_csv(io.BytesIO(official_content)).rename(columns={'path': 'file_path'})
    # Canonical clip IDs come from official paths.
    official['clip_id'] = official.file_path.map(lambda p: Path(p).stem)
    membership = ['clip_id', 'file_path', 'label']
    if len(official) != 1077 or official.label.value_counts().to_dict() != {0: 977, 1: 100}:
        raise ValueError('Unexpected official test split')
    peer = ROOT / 'outputs/kaggle_peer_audit/r3d18'
    rows, evidence, predictions = [], {}, {}
    specs = [
        ('U-Net++', ROOT/'outputs/kaggle_results/test_evaluation/unetpp_rd_002_test_results.zip',
         ROOT/'configs/unetpp_rd_002_evaluation.json', ROOT/'src/rd_ultrasound_ml'),
        ('R3D-18', peer/'test_results/r3d18_rd_001_test_results.zip',
         peer/'branch_source/configs/r3d18_rd_001_evaluation.json', peer/'branch_source/src/rd_ultrasound_ml'),
        ('R(2+1)D-18', ROOT/'outputs/kaggle_results/r2plus1d18_final_verification/download/r2plus1d18_rd_001_test_results.zip',
         ROOT/'configs/r2plus1d18_rd_001_evaluation.json', ROOT/'src/rd_ultrasound_ml'),
        ('U-Net++ + augmentation', Path(augmentation_archive).resolve(),
         ROOT/'configs/unetpp_rd_aug_001_evaluation.json', ROOT/'src/rd_ultrasound_ml')]
    for name, path, freeze_path, source_root in specs:
        files, audit = read_archive(path)
        load = lambda n: json.loads(files[n])
        freeze, result = load('evaluation_freeze.json'), load('test_metrics.json')
        if (freeze != read(freeze_path) or result['freeze_digest'] != digest(freeze)
                or freeze['threshold'] != .5 or freeze['test_accessed_at_freeze'] is not False
                or result['selected_epoch'] != freeze['selected_epoch']
                or result['checkpoint_sha256'] != freeze['checkpoint_sha256']
                or result['test_clips'] != 1077 or result['test_used'] is not True
                or result['training_performed'] is not False or result['threshold_tuned'] is not False
                or result['runtime'] != load('runtime.json')):
            raise ValueError(name + ': freeze or inference record mismatch')
        if 'training_runtime' in freeze and result['runtime'] != freeze['training_runtime']:
            raise ValueError(name + ': runtime drift')
        for filename, content in files.items():
            if filename.endswith('.py') and content != (source_root/filename).read_bytes():
                raise ValueError(name + ': source snapshot mismatch: ' + filename)
        for module, expected in freeze['training_implementation_sha256'].items():
            if hashlib.sha256(files[module+'.py']).hexdigest() != expected:
                raise ValueError(name + ': frozen training source mismatch')
        provenance = load('test_input_provenance.json')
        if provenance['test_csv_sha256'] != hashlib.sha256(official_content).hexdigest():
            raise ValueError(name + ': official test provenance mismatch')
        frame = pd.read_csv(io.BytesIO(files['test_predictions.csv']), dtype={'rd_probability': str})
        frame['rd_probability'] = frame.rd_probability.map(float)
        manifest = pd.read_csv(io.BytesIO(files['test_manifest.csv']))
        if (not frame[membership].equals(official[membership])
                or not manifest[membership].equals(official[membership]) or not frame.clip_id.is_unique):
            raise ValueError(name + ': test membership, order or labels mismatch')
        measured = metrics(frame)
        for key, value in measured.items():
            if not np.isclose(value, result['metrics'][key], rtol=0, atol=1e-12):
                raise ValueError(name + ': recomputed metric mismatch: ' + key)
        rows.append({'model': name, 'selected_epoch': freeze['selected_epoch'], **measured})
        predictions[name] = frame
        evidence[name] = {**audit, 'archive': str(path.relative_to(ROOT)),
            'checkpoint_sha256': freeze['checkpoint_sha256'], 'freeze_digest': digest(freeze),
            'test_input_provenance': provenance,
            'runtime': result['runtime'], 'inference_timing': result.get('inference_timing'),
            'predictions_verified': len(frame), 'metrics_recomputed': True}
    reference = evidence['U-Net++']['test_input_provenance']
    for name, record in evidence.items():
        for key in ('test_csv_sha256', 'prepared_manifest_sha256', 'preprocessing_identity'):
            if record['test_input_provenance'][key] != reference[key]:
                raise ValueError(name + ': reused test cache identity differs: ' + key)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output/'reused_test_metrics.csv', index=False)
    paired = official[membership].copy()
    for name, frame in predictions.items():
        paired[name+' probability'] = frame.rd_probability
        paired[name+' correct'] = frame.predicted_label == frame.label
    paired.to_csv(output/'paired_test_predictions.csv', index=False)
    base, aug = predictions['U-Net++'], predictions['U-Net++ + augmentation']
    b, a = base.predicted_label == base.label, aug.predicted_label == aug.label
    paired_counts = {'both_correct': int((b&a).sum()), 'both_wrong': int((~b&~a).sum()),
        'baseline_correct_augmentation_wrong': int((b&~a).sum()),
        'baseline_wrong_augmentation_correct': int((~b&a).sum())}
    (output/'verification.json').write_text(json.dumps({'status': 'passed',
        'benchmark': 'reused official test benchmark', 'threshold': .5,
        'model_inference_performed': False, 'sources': evidence,
        'augmentation_vs_baseline_paired': paired_counts, 'metrics': rows}, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for row in rows:
        name = row['model']
        f = predictions[name]
        x, y, _ = roc_curve(f.label, f.rd_probability)
        axes[0].plot(x, y, label=f"{name} (AUC {row['ROC_AUC']:.3f})")
        p, r, _ = precision_recall_curve(f.label, f.rd_probability)
        axes[1].step(r, p, where='post', label=f"{name} (AP {row['average_precision']:.3f})")
    axes[0].plot([0, 1], [0, 1], ':', color='gray', label='Chance')
    axes[1].axhline(100/1077, linestyle=':', color='gray', label='RD prevalence')
    axes[0].set(xlabel='False positive rate', ylabel='Sensitivity', title='ROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall')
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='lower left')
    fig.suptitle('Reused official test benchmark · 1,077 clips (100 RD) · frozen selections')
    for ext in ('png', 'pdf'):
        fig.savefig(output/('test_roc_precision_recall.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), layout='constrained')
    for ax, row in zip(axes, rows):
        counts = np.array([[row['TN'], row['FP']], [row['FN'], row['TP']]])
        ax.imshow(counts, cmap='Blues', vmin=0, vmax=977)
        for (i, j), value in np.ndenumerate(counts):
            ax.text(j, i, str(value), ha='center', va='center',
                    color='white' if value > 500 else 'black', fontsize=14)
        ax.set(xticks=[0, 1], xticklabels=['non-RD', 'RD'], yticks=[0, 1],
               yticklabels=['non-RD', 'RD'], xlabel='Model prediction',
               title=row['model'].replace(' + ', '\n+ '))
    axes[0].set_ylabel('Dataset label')
    fig.suptitle('Reused official test benchmark · threshold 0.5')
    for ext in ('png', 'pdf'):
        fig.savefig(output/('test_confusion_matrices.'+ext), dpi=180)
    plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
    print('Augmentation versus baseline:', paired_counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--augmentation-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.augmentation_archive, args.output)
