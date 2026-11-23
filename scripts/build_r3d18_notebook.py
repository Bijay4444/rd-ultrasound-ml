"""Build the standalone Kaggle GPU notebook for r3d18_rd_001_amp.

Reads the frozen config and tracked modules; writes only
notebooks/kaggle_r3d18_training.ipynb. U-Net++ configs/notebooks are untouched.
"""
import json
from pathlib import Path
import pprint
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/r3d18_rd_001_training.json'
NOTEBOOK = ROOT/'notebooks/kaggle_r3d18_training.ipynb'
MODULES = ['unetpp_baseline', 'unetpp_training', 'unetpp_data', 'unetpp_kaggle',
           'video_cache', 'comparison_export', 'r3d18_baseline']


def notebook():
    config = json.loads(CONFIG.read_text())
    cells = []

    def add(kind, ident, source):
        cell = {'cell_type': kind, 'id': ident, 'metadata': {},
                'source': dedent(source).lstrip('\n').splitlines(keepends=True)}
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n') + '\n'
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', 'r3d-intro', '''
# R3D-18 RD: Kaggle GPU training (`r3d18_rd_001_amp`)

Comparison model for the final milestone: torchvision `r3d_18(weights=None,
num_classes=1)` with only its first stem convolution changed to one grayscale
input channel. Inherited U-Net++ recipe: seed 42, 50 epochs, Adam 1.5e-5,
unweighted BCE, ReduceLROnPlateau, effective batch 48, threshold 0.5, highest
validation accuracy (earliest ties). No augmentation, pretrained weights,
class weighting or resampling.

Attach the **two completed U-Net++ CPU preparation outputs** (read-only). This
notebook never downloads videos and never opens the official test split. The
cache adapter validates the original preparation config against its pinned
digest, reruns every tensor/source/order check, and binds this run's own
training identity into resume checks.

Session 1 runs a disposable real-input precision/memory probe, then one full
epoch for timing (`R3D_EPOCHS_THIS_SESSION = 1`). That epoch starts from seed 42
and fresh weights and is epoch 1 of the fixed run; probe weights are discarded.
To continue, create a new saved job that also attaches the preceding successful
GPU output, and set `R3D_REQUIRE_PREVIOUS = True`
(`scripts/prepare_r3d18_job.py --session N` does this). Only GPU 0 is used.
A run stopped before 50 epochs is reported as partial.
''')
    add('code', 'kaggle-settings', '''
R3D_EPOCHS_THIS_SESSION = 1
R3D_REQUIRE_PREVIOUS = False
R3D_MAX_SESSION_SECONDS = 11 * 3600
''')
    add('code', 'r3d-runtime', '''
import gc
import hashlib
import json
import sys
import time
import types
import zipfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
import huggingface_hub
from IPython.display import display

if not Path('/kaggle/working').is_dir():
    raise RuntimeError('Submit this notebook to Kaggle; its remote environment runs the workload.')
torch.set_num_threads(2)
cv2.setNumThreads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
r3d_runtime = {'python': sys.version, 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
    'numpy': np.__version__, 'opencv': cv2.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
    'huggingface_hub': huggingface_hub.__version__, 'cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
    'cpu_torch_threads': torch.get_num_threads(), 'cudnn_benchmark': False,
    'matmul_tf32': False, 'cudnn_tf32': False}
print(r3d_runtime)
''')
    add('code', 'kaggle-config', 'R3D_CONFIG = ' + pprint.pformat(config, sort_dicts=False, width=100) + '\n')
    add('code', 'r3d-module-loader', '''
# In-memory package so the notebook needs no repository upload.
r3d_package = types.ModuleType('rd_ultrasound_ml')
r3d_package.__path__ = []
sys.modules['rd_ultrasound_ml'] = r3d_package
r3d_sources_embedded = {}
def r3d_load_module(name, source):
    module = types.ModuleType('rd_ultrasound_ml.' + name)
    module.__file__ = '<embedded ' + name + '.py>'
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    r3d_sources_embedded[name] = source
''')
    for name in MODULES:
        source = (ROOT/f'src/rd_ultrasound_ml/{name}.py').read_text(encoding='utf-8')
        add('code', 'kaggle-embed-' + name,
            'R3D_MODULE_SOURCE = ' + repr(source) + '\nr3d_load_module(' + repr(name) + ', R3D_MODULE_SOURCE)\n')
    add('code', 'r3d-imports', '''
from rd_ultrasound_ml.r3d18_baseline import R3D18RDClassifier, describe_model, gradcam_layer, GRADCAM_LAYER
from rd_ultrasound_ml.unetpp_training import (json_digest, file_digest, write_json_atomic,
    run_training, seed_training, binary_metrics)
from rd_ultrasound_ml.unetpp_data import CachedClips
from rd_ultrasound_ml.unetpp_kaggle import import_previous_run
from rd_ultrasound_ml.video_cache import load_cache_for_run
from rd_ultrasound_ml.comparison_export import probe_physical_batches, prediction_rows, verify_best_checkpoint
R3D_CONFIG['implementation_sha256'] = {name: hashlib.sha256(source.encode()).hexdigest()
    for name, source in r3d_sources_embedded.items()}
r3d_model_record = describe_model()
if r3d_model_record['parameter_count'] != R3D_CONFIG['model']['expected_parameter_count']:
    raise ValueError(f"Installed torchvision builds {r3d_model_record['parameter_count']} parameters; config expects "
                     f"{R3D_CONFIG['model']['expected_parameter_count']}. Inspect before training.")
print('Model:', r3d_model_record['builder'], '| parameters:', r3d_model_record['parameter_count'],
      '| Grad-CAM layer:', GRADCAM_LAYER)
''')
    add('code', 'r3d-gpu-inputs', '''
if not torch.cuda.is_available():
    raise RuntimeError('Select an available Kaggle CUDA GPU for this training notebook.')
r3d_input_root = Path('/kaggle/input')
r3d_train, r3d_validation, r3d_paths, r3d_fingerprint, r3d_cache_record = load_cache_for_run(
    r3d_input_root, R3D_CONFIG, report=lambda value: print(value, flush=True))
r3d_output = Path('/kaggle/working') / R3D_CONFIG['experiment_id']
r3d_output.mkdir(parents=True, exist_ok=True)
r3d_previous = import_previous_run(r3d_input_root, r3d_output, R3D_CONFIG, R3D_REQUIRE_PREVIOUS)
for name, value in [('config.json', R3D_CONFIG), ('runtime.json', r3d_runtime), ('model.json', r3d_model_record)]:
    path = r3d_output/name
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Saved {name} differs; inspect before continuing the fixed run.')
    write_json_atomic(path, value)
for name, source in r3d_sources_embedded.items():
    (r3d_output/(name+'.py')).write_text(source)
r3d_train.to_csv(r3d_output/'train_manifest.csv', index=False)
r3d_validation.to_csv(r3d_output/'validation_manifest.csv', index=False)
write_json_atomic(r3d_output/'input_fingerprint.json', {**r3d_cache_record, 'test_used': False})
r3d_train_dataset = CachedClips(r3d_paths[:len(r3d_train)], r3d_train['label'])
r3d_validation_dataset = CachedClips(r3d_paths[len(r3d_train):], r3d_validation['label'])
print('READY:', len(r3d_train_dataset), 'train,', len(r3d_validation_dataset), 'validation; no videos downloaded.')
print('Cache record:', r3d_cache_record)
print('Previous run:', r3d_previous)
''')
    add('code', 'r3d-gpu-probe', '''
# Disposable probe on one real training clip per class; weights are discarded.
seed_training(R3D_CONFIG['seed'])
r3d_physical = R3D_CONFIG['training']['physical_batch']
r3d_probe_indices = [int(np.flatnonzero(r3d_train['label'].to_numpy()==label)[0]) for label in [0, 1]]
r3d_probe_inputs = torch.stack([r3d_train_dataset[index][0] for index in r3d_probe_indices])
r3d_probe_labels = torch.tensor([0., 1.])
r3d_probe = probe_physical_batches(R3D18RDClassifier, r3d_probe_inputs, r3d_probe_labels,
    [batch for batch in (4, 2, 1) if batch <= r3d_physical], R3D_CONFIG['training']['precision'])
write_json_atomic(r3d_output/'precision_probe.json', {'probes': r3d_probe,
    'probe_clip_ids': r3d_train['clip_id'].iloc[r3d_probe_indices].tolist(), 'partition': 'train'})
for row in r3d_probe:
    print(row)
del r3d_probe_inputs, r3d_probe_labels
gc.collect()
torch.cuda.empty_cache()
if r3d_probe[0]['status'] != 'passed' or r3d_probe[0]['physical_batch'] != r3d_physical:
    raise RuntimeError(f'Configured physical batch {r3d_physical} did not pass. Coordinate a common smaller '
                       'physical batch with the R(2+1)D-18 owner and freeze a new config before training.')
''')
    add('code', 'r3d-gpu-train', '''
if not isinstance(R3D_EPOCHS_THIS_SESSION, int) or not 1 <= R3D_EPOCHS_THIS_SESSION <= 50:
    raise ValueError('Choose 1-50 epochs per saved job; start with 1 for real timing.')
r3d_session_start = time.perf_counter()
for r3d_session_epoch in range(R3D_EPOCHS_THIS_SESSION):
    torch.cuda.reset_peak_memory_stats()
    r3d_summary = run_training(R3D18RDClassifier, r3d_train_dataset, r3d_validation_dataset,
        R3D_CONFIG, r3d_output, r3d_fingerprint, r3d_runtime,
        device='cuda', max_epochs_this_call=1, report=lambda value: print(value, flush=True))
    r3d_history = json.loads((r3d_output/'history.json').read_text())
    pd.DataFrame(r3d_history).to_csv(r3d_output/'history.csv', index=False)
    r3d_memory_path = r3d_output/'epoch_memory.json'
    r3d_memory = json.loads(r3d_memory_path.read_text()) if r3d_memory_path.exists() else []
    r3d_memory.append({'epoch': r3d_history[-1]['epoch'], 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'seconds_this_segment': r3d_history[-1]['seconds_this_segment']})
    write_json_atomic(r3d_memory_path, r3d_memory)
    if r3d_summary['status'] == 'completed':
        break
    elapsed = time.perf_counter()-r3d_session_start
    last_seconds = r3d_history[-1]['seconds_this_segment']
    if elapsed+1.5*last_seconds+600 > R3D_MAX_SESSION_SECONDS:
        print('Returning normally to save partial output before the session budget; continue in a new saved job.')
        break
print('Status:', r3d_summary['status'], '| epochs:', r3d_summary['epochs_completed'], '/ 50')
print('Best epoch by validation accuracy:', r3d_summary['best_epoch'])
display(pd.DataFrame(r3d_history))
''')
    add('code', 'r3d-gpu-export', '''
r3d_threshold = R3D_CONFIG['evaluation']['threshold']
r3d_reload = verify_best_checkpoint(R3D18RDClassifier, r3d_output, r3d_validation_dataset, 'cuda',
    r3d_physical, R3D_CONFIG['training']['precision'], r3d_threshold)
write_json_atomic(r3d_output/'best_checkpoint_reload.json', r3d_reload)
r3d_best = json.loads((r3d_output/'best_validation_predictions.json').read_text())
pd.DataFrame(prediction_rows(r3d_validation, r3d_best['dataset_labels'], r3d_best['rd_probabilities'],
    r3d_threshold, 'validation')).to_csv(r3d_output/'best_validation_predictions.csv', index=False)
# Grad-CAM integration record: exact module path and feature shape on one real validation clip.
r3d_model = R3D18RDClassifier()
r3d_model.load_state_dict(torch.load(r3d_output/'best.pt', map_location='cpu', weights_only=True)['model_state'])
r3d_model.to('cuda').eval()
r3d_features = {}
r3d_hook = gradcam_layer(r3d_model).register_forward_hook(
    lambda module, inputs, output: r3d_features.__setitem__('shape', list(output.shape)))
with torch.no_grad():
    r3d_model(r3d_validation_dataset[0][0][None].to('cuda'))
r3d_hook.remove()
write_json_atomic(r3d_output/'gradcam_layer.json', {'module_path': GRADCAM_LAYER,
    'feature_shape_for_one_clip': r3d_features['shape'], 'input_shape': [1, 1, 96, 128, 128],
    'target': 'scalar RD logit (model output before sigmoid)', 'clip_id': r3d_validation['clip_id'].iloc[0]})
del r3d_model
torch.cuda.empty_cache()
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r3d_frame = pd.DataFrame(json.loads((r3d_output/'history.json').read_text()))
    r3d_figure, r3d_axes = plt.subplots(1, 2, figsize=(11, 4))
    r3d_axes[0].plot(r3d_frame['epoch'], r3d_frame['train_loss'], label='train BCE')
    r3d_axes[0].plot(r3d_frame['epoch'], r3d_frame['validation_loss'], label='validation BCE')
    for metric in ['accuracy', 'sensitivity', 'specificity']:
        r3d_axes[1].plot(r3d_frame['epoch'], r3d_frame[metric], label='validation ' + metric)
    for axis in r3d_axes:
        axis.set_xlabel('epoch')
        axis.legend()
    r3d_figure.tight_layout()
    r3d_figure.savefig(r3d_output/'training_curves.png', dpi=120)
    plt.close(r3d_figure)
except ImportError:
    print('matplotlib unavailable; history.csv contains every curve value.')
r3d_hashes = {path.name: file_digest(path) for path in sorted(r3d_output.iterdir())
    if path.is_file() and path.suffix in {'.json', '.csv', '.pt', '.py', '.png'} and path.name != 'artifact_sha256.json'}
write_json_atomic(r3d_output/'artifact_sha256.json', r3d_hashes)
r3d_zip = r3d_output.parent/(R3D_CONFIG['experiment_id'] + '_results.zip')
with zipfile.ZipFile(r3d_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in [*r3d_hashes, 'artifact_sha256.json']:
        archive.write(r3d_output/name, arcname=f'{r3d_output.name}/{name}')
print('Reload check:', r3d_reload)
print('Results ZIP:', r3d_zip)
print('A partial job is not a completed 50-epoch run. No test evaluation was run.')
''')
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'kaggle': {'isGpuEnabled': True, 'isInternetEnabled': False}},
        'nbformat': 4, 'nbformat_minor': 5}


def build():
    NOTEBOOK.write_text(json.dumps(notebook(), indent=1) + '\n', encoding='utf-8')
    return NOTEBOOK


if __name__ == '__main__':
    print(build())
