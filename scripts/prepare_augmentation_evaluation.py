"""Freeze the selected augmented model and prepare its gated, single test evaluation.

Preparation works without team confirmation. A runnable submission bundle is
created only with --confirm-team-freeze after the team decision is recorded.
Does not submit jobs, open test data, select a threshold or change training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pprint

from build_augmentation_notebook import MODULES as TRAINING_MODULES
from prepare_augmentation_job import DOCKER_IMAGE

ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / 'configs/unetpp_rd_aug_001_evaluation.json'
NOTEBOOK_PATH = ROOT / 'notebooks/kaggle_unetpp_augmentation_evaluation.ipynb'
MODULES = [*TRAINING_MODULES, 'unetpp_evaluation', 'comparison_evaluation']
SOURCE = 'bijaykhadka1710/unetpp-rd-aug-001-train-03'
SLUG = 'unetpp-rd-aug-001-test-evaluation'


def make_freeze(training, confirmed=False, note=None, existing=None):
    training = Path(training)
    def read(name):
        return json.loads((training / name).read_text())
    summary, config, identity = read('summary.json'), read('config.json'), read('input_fingerprint.json')
    history = read('history.json')
    if (summary['status'] != 'completed' or summary['epochs_completed'] != 50 or
            summary['best_epoch'] != max(history, key=lambda r: r['accuracy'])['epoch'] or
            summary['test_used'] is not False):
        raise ValueError('Require completed training with the original selected checkpoint')
    test_reference = json.loads((ROOT / 'configs/r2plus1d18_rd_001_evaluation.json').read_text())
    selection = {'training_experiment_id': config['experiment_id'], 'selected_epoch': summary['best_epoch'],
        'checkpoint_sha256': hashlib.sha256((training / 'best.pt').read_bytes()).hexdigest(),
        'training_config_digest': summary['config_digest'], 'training_run_fingerprint': summary['data_fingerprint']}
    if existing and any(existing[k] != value for k, value in selection.items()):
        raise ValueError('Previously frozen model identity differs; preserve it before another experiment')
    freeze = {'schema_version': 1, 'evaluation_id': 'unetpp_rd_aug_001_test',
        'model_label': 'U-Net++ + training augmentation',
        'frozen_at_utc': (existing or {}).get('frozen_at_utc', datetime.now(timezone.utc).isoformat()),
        'team_freeze_confirmed': confirmed or (existing or {}).get('team_freeze_confirmed', False),
        'team_freeze_note': note or (existing or {}).get('team_freeze_note',
            'Pending confirmation of the final peer artifact/model choices. Augmented epoch 32 and threshold 0.5 are fixed from validation; no augmented-model test inference has been run.'),
        **selection, 'training_source': SOURCE, 'training_source_version': 1,
        'training_epochs_completed': 50, 'selection_rule': config['training']['checkpoint_selection'],
        'checkpoint_file': 'best.pt', 'training_implementation_sha256': config['implementation_sha256'],
        'training_runtime': read('runtime.json'), 'dataset_revision': config['dataset_revision'],
        'task': config['task'], 'positive_class': 'rd', 'threshold': .5,
        'precision': config['training']['precision'], 'physical_batch': config['training']['physical_batch'],
        'preprocessing_identity': identity['preprocessing_identity'], 'test_cache': test_reference['test_cache'],
        'test_clips': 1077, 'test_label_counts': {'0': 977, '1': 100},
        'test_accessed_at_freeze': False, 'benchmark_label': 'reused official test benchmark',
        'test_policy': 'single frozen-checkpoint evaluation; clean inputs; no augmentation, optimizer, calibration, threshold search or checkpoint selection',
        'reference_test_rows': test_reference['reference_test_rows'],
        'disclosures': [
            'Both U-Net++ runs completed 50 epochs with matched settings/runtime; selected augmented epoch is 32, baseline epoch is 48.',
            'Validation did not improve with this augmentation policy; retain the negative result and every benchmark result.',
            'Training-only transforms are disabled for validation/test. Cached preprocessing is unchanged.',
            'Official test scores already exist at project level. This is benchmark reuse, not an untouched holdout.',
            'One seed; possible same-patient overlap in clip-level splits; no clinical or unseen-patient validation claim.']}
    return freeze


def notebook(freeze):
    shared = {c['id']: ''.join(c['source']) for c in json.loads(
        (ROOT / 'notebooks/kaggle_unetpp_training.ipynb').read_text())['cells']}
    cells = []
    def add(kind, ident, source):
        cell = {'cell_type': kind, 'id': ident, 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', 'intro', '''# U-Net++ augmentation: frozen benchmark evaluation

Requires confirmation of final team model choices. Otherwise stop before any
test data access. Freeze the selected epoch 32, threshold 0.5, exact weights,
training config/source/runtime and original preprocessing. Attach only the final
augmented training output (session 03, version 1) and original saved test cache.
Use the exact baseline Docker image and clean inputs. No augmentation, training,
threshold search or selection occurs. Report as a reused official test benchmark.
''')
    add('code', 'runtime', shared['kaggle-runtime'])
    add('code', 'freeze', 'AUG_FREEZE = ' + pprint.pformat(freeze, sort_dicts=False, width=100) + '\n')
    add('code', 'module-loader', shared['kaggle-module-loader'])
    for name in MODULES:
        add('code', 'embed-' + name, 'unetpp_load_module(' + repr(name) + ', ' +
            repr((ROOT / f'src/rd_ultrasound_ml/{name}.py').read_text()) + ')\n')
    add('code', 'verify-before-test-access', '''from rd_ultrasound_ml.unetpp_baseline import UNetPPRDClassifier
from rd_ultrasound_ml.unetpp_training import file_digest, json_digest, write_json_atomic
from rd_ultrasound_ml.comparison_evaluation import find_comparison_training, load_reused_test_cache, evaluate_comparison_model
if AUG_FREEZE.get('team_freeze_confirmed') is not True:
    raise RuntimeError('Final team choices are not confirmed; no test data opened.')
for name, digest in AUG_FREEZE['training_implementation_sha256'].items():
    if hashlib.sha256(unetpp_sources_embedded[name].encode()).hexdigest() != digest:
        raise ValueError('Embedded training source differs from freeze: ' + name)
aug_training = find_comparison_training(Path('/kaggle/input'), AUG_FREEZE)
if unetpp_runtime != AUG_FREEZE['training_runtime'] or json.loads((aug_training / 'runtime.json').read_text()) != unetpp_runtime:
    raise ValueError('Runtime differs from frozen training; no test data opened.')
if not torch.cuda.is_available():
    raise RuntimeError('A CUDA GPU is required for the frozen inference configuration.')
''')
    add('code', 'verified-test-inputs', '''aug_test, aug_dataset, aug_input_record = load_reused_test_cache(
    Path('/kaggle/input'), aug_training, AUG_FREEZE, report=lambda value: print(value, flush=True))
aug_output = Path('/kaggle/working') / AUG_FREEZE['evaluation_id']
aug_output.mkdir(parents=True, exist_ok=True)
write_json_atomic(aug_output / 'runtime.json', unetpp_runtime)
write_json_atomic(aug_output / 'test_input_provenance.json', aug_input_record)
for name, source in unetpp_sources_embedded.items():
    (aug_output / (name + '.py')).write_text(source)
''')
    add('code', 'single-inference', '''aug_result = evaluate_comparison_model(UNetPPRDClassifier, aug_dataset, aug_test,
    aug_training, AUG_FREEZE, aug_output, unetpp_runtime, device='cuda', report=lambda value: print(value, flush=True))
print('REUSED OFFICIAL TEST BENCHMARK:', aug_result['metrics'])
print('Inference timing:', aug_result['inference_timing'])
''')
    add('code', 'export', '''aug_hashes = {p.name: file_digest(p) for p in sorted(aug_output.iterdir())
    if p.is_file() and p.name != 'artifact_sha256.json'}
write_json_atomic(aug_output / 'artifact_sha256.json', aug_hashes)
with zipfile.ZipFile(aug_output.parent / (AUG_FREEZE['evaluation_id'] + '_results.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in [*aug_hashes, 'artifact_sha256.json']:
        archive.write(aug_output / name, arcname=aug_output.name + '/' + name)
print('Single frozen evaluation complete; no training or tuning performed.')
''')
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'kaggle': {'isGpuEnabled': True, 'isInternetEnabled': False}},
        'nbformat': 4, 'nbformat_minor': 5}


def write_bundle(freeze, output_root):
    if freeze.get('team_freeze_confirmed') is not True:
        raise ValueError('Confirm the final team choices before preparing a runnable test bundle')
    output = Path(output_root) / SLUG
    metadata = {'id': 'bijaykhadka1710/' + SLUG, 'title': 'UNetpp Rd Aug 001 Test Evaluation',
        'code_file': NOTEBOOK_PATH.name, 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': True, 'enable_internet': False,
        'machine_shape': 'NvidiaTeslaT4', 'docker_image': DOCKER_IMAGE,
        'dataset_sources': [], 'competition_sources': [], 'kernel_sources': [SOURCE + '/1',
            f"{freeze['test_cache']['source']}/{freeze['test_cache']['source_version']}"]}
    files = {'kernel-metadata.json': metadata, NOTEBOOK_PATH.name: notebook(freeze)}
    for name, value in files.items():
        target = output / name
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError('Existing bundle differs; preserve it and use a fresh output root')
    output.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        (output / name).write_text(json.dumps(value, indent=1) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=ROOT / 'outputs/augmentation/training_pipeline/results/session_03_v1/verified/unetpp_rd_aug_001_amp')
    parser.add_argument('--confirm-team-freeze', action='store_true')
    parser.add_argument('--confirmation-note')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/kaggle_jobs')
    args = parser.parse_args()
    if args.confirm_team_freeze and not args.confirmation_note:
        parser.error('Record the actual team confirmation in --confirmation-note')
    existing = json.loads(FREEZE_PATH.read_text()) if FREEZE_PATH.exists() else None
    freeze = make_freeze(args.training, args.confirm_team_freeze, args.confirmation_note, existing)
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2) + '\n')
    NOTEBOOK_PATH.write_text(json.dumps(notebook(freeze), indent=1) + '\n')
    print('Checkpoint freeze and notebook prepared:', FREEZE_PATH, NOTEBOOK_PATH)
    if freeze['team_freeze_confirmed']:
        print('Private evaluation bundle:', write_bundle(freeze, args.output_root))
    else:
        print('Team/peer confirmation pending: no runnable bundle or submission; no test data opened.')
