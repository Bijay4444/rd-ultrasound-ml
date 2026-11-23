"""Build an independent augmentation notebook; never rewrites baseline artifacts."""
import json
from pathlib import Path
import pprint
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
MODULES = ['unetpp_baseline', 'unetpp_training', 'unetpp_data', 'unetpp_kaggle',
           'video_cache', 'comparison_export', 'video_augmentation', 'augmentation_preview']
FILENAME = 'kaggle_unetpp_augmentation.ipynb'


def notebook():
    config = json.loads((ROOT / 'configs/unetpp_rd_aug_001_training.json').read_text())
    original = json.loads((ROOT / 'notebooks/kaggle_unetpp_training.ipynb').read_text())
    shared = {cell['id']: ''.join(cell['source']) for cell in original['cells']}
    cells = []
    def add(kind, ident, source):
        cell = {'cell_type': kind, 'id': ident, 'metadata': {},
            'source': dedent(source).lstrip('\n').splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', 'intro', '''
# U-Net++: training-only augmentation (`unetpp_rd_aug_001_amp`)

Separate from-scratch 50-epoch experiment with the baseline architecture,
seed, optimizer, loss, physical/effective batches, and selection rule preserved.
Only training views change. Attach the two version-pinned preparation outputs;
never attach a baseline checkpoint. No test CSV is opened. Cache checks remain.

The initial saved GPU job runs a disposable real-input probe and one full epoch
for timing. Further epochs require sufficient measured GPU budget. Continuations
attach this experiment's preceding successful saved output. Views are keyed by
seed/epoch/clip ID and replay across interruptions. Validation remains clean.
A partial run is not a 50-epoch result. Eight fixed training previews use the
exact policy; boundary stress examples are labeled separately. These establish
engineering behavior, not clinical invariance.
''')
    add('code', 'kaggle-settings', '''
AUG_EPOCHS_THIS_SESSION = 1
AUG_REQUIRE_PREVIOUS = False
AUG_MAX_SESSION_SECONDS = 6 * 3600
''')
    add('code', 'runtime', shared['kaggle-runtime'])
    add('code', 'config', 'AUG_CONFIG = ' + pprint.pformat(config, sort_dicts=False, width=100) + '\n')
    add('code', 'module-loader', shared['kaggle-module-loader'])
    for name in MODULES:
        source = (ROOT / f'src/rd_ultrasound_ml/{name}.py').read_text()
        add('code', 'embed-' + name, 'unetpp_load_module(' + repr(name) + ', ' + repr(source) + ')\n')
    add('code', 'imports-and-identity', '''
from rd_ultrasound_ml.unetpp_baseline import UNetPPRDClassifier
from rd_ultrasound_ml.unetpp_training import (json_digest, file_digest, write_json_atomic,
    probe_training_precision, seed_training)
from rd_ultrasound_ml.unetpp_data import CachedClips
from rd_ultrasound_ml.unetpp_kaggle import import_previous_run
from rd_ultrasound_ml.video_cache import load_cache_for_run
from rd_ultrasound_ml.video_augmentation import TrainingAugmentedClips, run_augmented_training
from rd_ultrasound_ml.augmentation_preview import save_preview
from rd_ultrasound_ml.comparison_export import prediction_rows, verify_best_checkpoint
AUG_CONFIG['implementation_sha256'] = {name: hashlib.sha256(source.encode()).hexdigest()
    for name, source in unetpp_sources_embedded.items()}
for name, digest in AUG_CONFIG['reference_implementation_sha256'].items():
    if AUG_CONFIG['implementation_sha256'][name] != digest:
        raise ValueError('Frozen baseline source changed: ' + name)
if json_digest(AUG_CONFIG['training_augmentation']) != AUG_CONFIG['preview']['policy_digest']:
    raise ValueError('Training policy differs from the reviewed preview')
for key, expected in AUG_CONFIG['reference_runtime'].items():
    if unetpp_runtime.get(key) != expected:
        raise RuntimeError(f'Runtime {key} differs from baseline: {unetpp_runtime.get(key)!r} vs {expected!r}; document before changing the comparison')
''')
    add('code', 'inputs', '''
if not torch.cuda.is_available():
    raise RuntimeError('Select a Kaggle CUDA GPU for the real training probe.')
aug_train, aug_validation, aug_paths, aug_fingerprint, aug_cache_record = load_cache_for_run(
    Path('/kaggle/input'), AUG_CONFIG, report=lambda value: print(value, flush=True))
aug_output = Path('/kaggle/working/unetpp_rd_aug_001_amp')
aug_output.mkdir(parents=True, exist_ok=True)
aug_previous = import_previous_run(Path('/kaggle/input'), aug_output, AUG_CONFIG, AUG_REQUIRE_PREVIOUS)
for name, value in [('config.json', AUG_CONFIG), ('runtime.json', unetpp_runtime)]:
    path = aug_output / name
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Saved configuration/runtime differs; retain the previous run')
    write_json_atomic(path, value)
for name, source in unetpp_sources_embedded.items():
    (aug_output / (name + '.py')).write_text(source)
aug_train.to_csv(aug_output / 'train_manifest.csv', index=False)
aug_validation.to_csv(aug_output / 'validation_manifest.csv', index=False)
write_json_atomic(aug_output / 'input_fingerprint.json', {**aug_cache_record, 'test_used': False})
aug_clean_train = CachedClips(aug_paths[:len(aug_train)], aug_train['label'])
aug_validation_dataset = CachedClips(aug_paths[len(aug_train):], aug_validation['label'])
aug_train_dataset = TrainingAugmentedClips(aug_clean_train, aug_train['clip_id'],
    AUG_CONFIG['training_augmentation'], AUG_CONFIG['seed'])
print('READY:', len(aug_train_dataset), 'augmented train;', len(aug_validation_dataset), 'clean validation')
print('Resume source:', aug_previous)
''')
    add('code', 'training-previews', '''
aug_preview = save_preview(aug_clean_train, aug_train, AUG_CONFIG['preview']['clip_ids'],
    AUG_CONFIG['training_augmentation'], AUG_CONFIG['seed'], aug_output / 'previews')
print('Training-only previews:', aug_output / 'previews/original_vs_augmented.png')
''')
    add('code', 'resource-probe', '''
seed_training(AUG_CONFIG['seed'])
aug_probe_indices = [int(np.flatnonzero(aug_train['label'].to_numpy() == label)[0]) for label in [0, 1]]
aug_probe_inputs = torch.stack([aug_train_dataset[index][0] for index in aug_probe_indices])
aug_probe = probe_training_precision(UNetPPRDClassifier, aug_probe_inputs, torch.tensor([0., 1.]),
    AUG_CONFIG['training']['physical_batch'], AUG_CONFIG['training']['precision'])
write_json_atomic(aug_output / 'precision_probe.json', aug_probe)
print(aug_probe)
del aug_probe_inputs
gc.collect()
torch.cuda.empty_cache()
''')
    add('code', 'training', '''
if type(AUG_EPOCHS_THIS_SESSION) is not int or not 1 <= AUG_EPOCHS_THIS_SESSION <= 50:
    raise ValueError('Choose 1-50 epochs for this bounded saved session')
aug_start = time.perf_counter()
for aug_session_epoch in range(AUG_EPOCHS_THIS_SESSION):
    aug_summary = run_augmented_training(UNetPPRDClassifier, aug_train_dataset,
        aug_validation_dataset, AUG_CONFIG, aug_output, aug_fingerprint, unetpp_runtime,
        device='cuda', report=lambda value: print(value, flush=True))
    aug_history = json.loads((aug_output / 'history.json').read_text())
    pd.DataFrame(aug_history).to_csv(aug_output / 'history.csv', index=False)
    if aug_summary['status'] == 'completed':
        break
    elapsed = time.perf_counter() - aug_start
    if elapsed + 1.5 * aug_history[-1]['seconds_this_segment'] + 300 > AUG_MAX_SESSION_SECONDS:
        print('Saving normally before session limit; continue in another saved job.')
        break
write_json_atomic(aug_output / 'budget_projection.json', {
    'epochs_completed': aug_summary['epochs_completed'],
    'last_epoch_seconds': aug_history[-1]['seconds_this_segment'],
    'remaining_epochs': 50 - aug_summary['epochs_completed'],
    'estimated_remaining_training_seconds': (50 - aug_summary['epochs_completed']) * aug_history[-1]['seconds_this_segment'],
    'excludes': 'cache checks, probes, restarts, exports and queue time', 'test_used': False})
print('Status:', aug_summary['status'], '| epochs:', aug_summary['epochs_completed'], '/ 50')
display(pd.DataFrame(aug_history))
''')
    add('code', 'validation-exports', '''
aug_reload = verify_best_checkpoint(UNetPPRDClassifier, aug_output, aug_validation_dataset,
    'cuda', AUG_CONFIG['training']['physical_batch'], AUG_CONFIG['training']['precision'])
write_json_atomic(aug_output / 'best_checkpoint_reload.json', aug_reload)
for aug_record in sorted(aug_output.glob('validation_epoch_*.json')):
    aug_prediction = json.loads(aug_record.read_text())
    pd.DataFrame(prediction_rows(aug_validation, aug_prediction['dataset_labels'],
        aug_prediction['rd_probabilities'], .5, 'validation')).to_csv(aug_record.with_suffix('.csv'), index=False)
aug_prediction = json.loads((aug_output / 'best_validation_predictions.json').read_text())
pd.DataFrame(prediction_rows(aug_validation, aug_prediction['dataset_labels'],
    aug_prediction['rd_probabilities'], .5, 'validation')).to_csv(aug_output / 'best_validation_predictions.csv', index=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
aug_frame = pd.DataFrame(aug_history)
aug_figure, aug_axes = plt.subplots(1, 2, figsize=(10, 4))
aug_axes[0].plot(aug_frame['epoch'], aug_frame['train_loss'], label='train BCE, augmented')
aug_axes[0].plot(aug_frame['epoch'], aug_frame['validation_loss'], label='validation BCE, clean')
aug_axes[1].plot(aug_frame['epoch'], aug_frame['sensitivity'], label='validation sensitivity')
aug_axes[1].plot(aug_frame['epoch'], aug_frame['specificity'], label='validation specificity')
for axis in aug_axes:
    axis.set_xlabel('Epoch'); axis.legend()
aug_figure.tight_layout(); aug_figure.savefig(aug_output / 'learning_curves.png', dpi=150); plt.close(aug_figure)
''')
    add('code', 'export', '''
aug_hashes = {str(path.relative_to(aug_output)): file_digest(path)
    for path in sorted(aug_output.rglob('*')) if path.is_file() and path.name != 'artifact_sha256.json'}
write_json_atomic(aug_output / 'artifact_sha256.json', aug_hashes)
aug_zip = aug_output.parent / 'unetpp_rd_aug_001_amp_results.zip'
with zipfile.ZipFile(aug_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in [*aug_hashes, 'artifact_sha256.json']:
        archive.write(aug_output / name, arcname=aug_output.name + '/' + name)
print('Results:', aug_zip, '| No final-test evaluation was run.')
''')
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'kaggle': {'isGpuEnabled': True, 'isInternetEnabled': False}},
        'nbformat': 4, 'nbformat_minor': 5}


if __name__ == '__main__':
    output = ROOT / 'notebooks' / FILENAME
    output.write_text(json.dumps(notebook(), indent=1) + '\n')
    print(output)
