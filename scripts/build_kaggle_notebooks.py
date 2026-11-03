"""Build standalone Kaggle notebooks from the project's exact tracked modules."""
import json
from pathlib import Path
import pprint
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def build():
    config = json.loads((ROOT/'configs/unetpp_rd_001_training.json').read_text())
    config.update(experiment_id='unetpp_rd_002_amp', platform='kaggle')
    config['disclosures'] += [
        'Kaggle platform: two saved CPU-preparation outputs preserve all float32 inputs; GPU jobs attach them read-only.',
        'Only CUDA device 0 is used, including on a T4 x2 allocation; this is not distributed training.',
        'Training proceeds in saved bounded notebook jobs; completed prior output must be attached to continue.']
    (ROOT/'configs/unetpp_rd_002_training.json').write_text(json.dumps(config, indent=2)+'\n')
    old = json.loads((ROOT/'notebooks/unetpp_training.ipynb').read_text())
    old_cells = {c['id']: ''.join(c['source']) for c in old['cells']}
    modules = ['unetpp_baseline','unetpp_training','unetpp_data','unetpp_kaggle']
    def notebook(mode):
        cells=[]
        def add(kind, ident, source):
            c={'cell_type':kind,'id':ident,'metadata':{},'source':dedent(source).lstrip('\n').splitlines(keepends=True)}
            if c['source']: c['source'][-1]=c['source'][-1].rstrip('\n')+'\n'
            if kind=='code':c.update(execution_count=None,outputs=[])
            cells.append(c)
        add('markdown','kaggle-intro', '''
# UNet++ RD: Kaggle CPU preparation

Run this notebook twice as **two separate private saved CPU jobs**, shard 0 and
shard 1. Use `python3 scripts/prepare_kaggle_job.py --help` from VS Code to create
both bundles. No GPU is used. Each job prepares exactly half of all 4,304 official
train/validation clips, preserving float32 `[1,96,128,128]` inputs. Each output is
about 13.54 GB, below Kaggle's documented 20 GB saved-output limit. Raw clips
are temporary and excluded from saved outputs. No test CSV is opened.

Wait for each saved run to finish successfully. Attach both completed outputs
to the separate GPU training notebook. A stopped interactive session alone does
not guarantee that its output files are saved. Browser route: upload this
notebook, choose Accelerator **None**, enable Internet, set the shard index, then
**Save Version → Save & Run All**. The two runs must have different notebook names.
''' if mode=='prepare' else '''
# UNet++ RD: Kaggle GPU training

Attach both successfully saved CPU-preparation outputs before enabling a GPU.
This notebook never downloads or preprocesses full videos. It checks attached
input hashes, runs a disposable mixed-precision GPU probe, and starts one full
epoch by default. All 3,873 training and 431 validation inputs remain unchanged.
The fixed target is 50 epochs; a shorter saved job is explicitly partial.

Use **Save Version → Save & Run All** (or the official CLI) to retain outputs.
For continuation create a new job, attach both CPU outputs **and the preceding
successful GPU output**, and set `UNETPP_REQUIRE_PREVIOUS = True`. The helper
`prepare_kaggle_job.py train --session 2` generates that dependency automatically.
Do not reset or rerun preparation for a new GPU job. Only GPU 0 is used, even
if Kaggle allocates two GPUs. No final-test evaluation is performed here.

Checkpoints written to `/kaggle/working` protect work inside the current runtime;
only successfully saved output is a durable handoff to the next run. Unexpected
termination can lose the current unsaved job. Bounded jobs return normally and
export results before the platform time limit; free GPU availability is not guaranteed.
''')
        add('code','kaggle-settings', '''
UNETPP_SHARD_INDEX = 0
''' if mode=='prepare' else '''
UNETPP_EPOCHS_THIS_SESSION = 1
UNETPP_REQUIRE_PREVIOUS = False
UNETPP_MAX_SESSION_SECONDS = 6 * 3600
''')
        add('code','kaggle-runtime','''
import gc
import hashlib
import json
import shutil
import sys
import time
import types
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
import huggingface_hub
from huggingface_hub import HfApi
from IPython.display import display

if not Path('/kaggle/working').is_dir():
    raise RuntimeError('Submit this notebook to Kaggle; its remote environment runs the workload.')
torch.set_num_threads(2)
cv2.setNumThreads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
unetpp_runtime = {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
    'opencv': cv2.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
    'huggingface_hub': huggingface_hub.__version__, 'cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'cpu_torch_threads': torch.get_num_threads(), 'cudnn_benchmark': False,
    'matmul_tf32': False, 'cudnn_tf32': False}
print(unetpp_runtime)
''')
        add('code','kaggle-config','UNETPP_CONFIG = '+pprint.pformat(config,sort_dicts=False,width=100)+'\n')
        add('code','kaggle-module-loader',old_cells['training-module-loader'])
        for name in modules:
            source=(ROOT/f'src/rd_ultrasound_ml/{name}.py').read_text()
            add('code','kaggle-embed-'+name,'UNETPP_MODULE_SOURCE = '+repr(source)+'\nunetpp_load_module('+repr(name)+', UNETPP_MODULE_SOURCE)\n')
        add('code','kaggle-imports','''
from rd_ultrasound_ml.unetpp_baseline import UNetPPRDClassifier
from rd_ultrasound_ml.unetpp_training import (json_digest, file_digest, write_json_atomic,
    run_training, probe_training_precision, seed_training, binary_metrics)
from rd_ultrasound_ml.unetpp_data import (record_video_sources, prepare_cached_clip, CachedClips)
from rd_ultrasound_ml.unetpp_kaggle import (shard_indices, read_official_inputs, load_prepared_shards, import_previous_run)
UNETPP_CONFIG['implementation_sha256'] = {name: hashlib.sha256(source.encode()).hexdigest()
    for name, source in unetpp_sources_embedded.items()}
''')
        if mode=='prepare':
            add('code','kaggle-cpu-storage','''
if torch.cuda.is_available():
    raise RuntimeError('CPU preparation must run with Accelerator None to preserve GPU quota.')
unetpp_output = Path('/kaggle/working') / f'unetpp_inputs_{UNETPP_SHARD_INDEX:02d}'
unetpp_output.mkdir(parents=True, exist_ok=True)
write_json_atomic(unetpp_output/'config.json', UNETPP_CONFIG)
write_json_atomic(unetpp_output/'runtime.json', unetpp_runtime)
print('CPU shard:', UNETPP_SHARD_INDEX, '| saved output:', unetpp_output)
''')
            add('code','kaggle-download-helper',old_cells['training-download-helper'])
            add('code','kaggle-cpu-partitions','''
unetpp_release = 'https://zenodo.org/records/18644370'
download_checked(f'{unetpp_release}/files/erdes_metadata.csv?download=1',
    unetpp_output/'erdes_metadata.csv', '95828fb5bdffdbf8173b5bcfc99e8148', algorithm='md5')
download_checked(f'{unetpp_release}/files/splits.zip?download=1',
    unetpp_output/'splits.zip', 'c5630e3f056ce358dcd384aaed946e64', algorithm='md5')
unetpp_train, unetpp_validation = read_official_inputs(unetpp_output)
unetpp_all = pd.concat([unetpp_train, unetpp_validation], ignore_index=True)
unetpp_positions = shard_indices(len(unetpp_all), UNETPP_SHARD_INDEX)
unetpp_selected = unetpp_all.iloc[unetpp_positions]
unetpp_source_catalog = record_video_sources(unetpp_selected, UNETPP_CONFIG['dataset_revision'],
    unetpp_output/'video_sources.json', HfApi(token=False), report=lambda value: print(value, flush=True))
unetpp_preprocessing_identity = {'preprocessing': UNETPP_CONFIG['preprocessing'],
    'implementation': UNETPP_CONFIG['implementation_sha256']['unetpp_baseline'],
    'torch': str(torch.__version__), 'opencv': cv2.__version__, 'numpy': np.__version__}
unetpp_signature = json_digest(unetpp_preprocessing_identity)
unetpp_cache = unetpp_output/'arrays'
unetpp_required = len(unetpp_positions)*(96*128*128*4+128)+500_000_000
unetpp_existing = sum(path.stat().st_size for path in unetpp_cache.glob('*.npy')) if unetpp_cache.exists() else 0
if unetpp_required > 19_000_000_000 or shutil.disk_usage(unetpp_output).free < max(0, unetpp_required-unetpp_existing):
    raise RuntimeError('Insufficient saved-output space for this CPU shard.')
print('Selected clips:', len(unetpp_positions), '| float32 array GB:', round(len(unetpp_positions)*96*128*128*4/1e9,2))
''')
            add('code','kaggle-cpu-prepare','''
unetpp_entries = []
unetpp_start = time.perf_counter()
# Raw data never accumulate in saved outputs. This task owns the temporary directory.
with TemporaryDirectory(prefix='unetpp_raw_', dir='/tmp') as unetpp_raw_directory:
    for count, position in enumerate(unetpp_positions, 1):
        row = next(unetpp_all.iloc[[position]].itertuples(index=False))
        source = unetpp_source_catalog[row.file_path]
        path, check = prepare_cached_clip(row, source, unetpp_raw_directory,
            unetpp_cache, unetpp_signature, download_checked)
        raw_path = Path(unetpp_raw_directory)/'clips'/row.file_path
        if raw_path.exists():
            raw_path.unlink()  # This temporary copy was created solely by this preparation job.
        unetpp_entries.append({'global_index': position, 'clip_id': row.clip_id, 'label': int(row.label),
            'array': str(path.relative_to(unetpp_output)), 'source': source, 'check': check})
        if count == 1 or count % 10 == 0 or count == len(unetpp_positions):
            print(f'CPU shard {UNETPP_SHARD_INDEX}: {count}/{len(unetpp_positions)} prepared', flush=True)
            write_json_atomic(unetpp_output/'preparation_progress.json',
                {'prepared_clips': count, 'selected_clips': len(unetpp_positions), 'complete': False})
write_json_atomic(unetpp_output/'prepared_manifest.json', {'schema_version': 1, 'complete': True,
    'shard_index': UNETPP_SHARD_INDEX, 'shard_count': 2, 'config_digest': json_digest(UNETPP_CONFIG),
    'preprocessing_identity': unetpp_preprocessing_identity, 'entries': unetpp_entries, 'test_used': False})
write_json_atomic(unetpp_output/'preparation_summary.json', {'complete': True,
    'shard_index': UNETPP_SHARD_INDEX, 'shard_clips': len(unetpp_entries),
    'total_train_clips': len(unetpp_train), 'total_validation_clips': len(unetpp_validation),
    'seconds': time.perf_counter()-unetpp_start,
    'output_bytes': sum(path.stat().st_size for path in unetpp_output.rglob('*') if path.is_file()),
    'dtype': 'float32', 'gpu_used': False, 'test_used': False})
print('CPU preparation COMPLETE. Wait for Kaggle to finish saving this output before using it as input.')
print('Saved output:', unetpp_output)
''')
        else:
            add('code','kaggle-gpu-inputs','''
if not torch.cuda.is_available():
    raise RuntimeError('Select an available Kaggle CUDA GPU for this training notebook.')
unetpp_input_root = Path('/kaggle/input')
unetpp_train, unetpp_validation, unetpp_paths, unetpp_data_fingerprint, unetpp_preprocessing_identity = load_prepared_shards(
    unetpp_input_root, UNETPP_CONFIG, report=lambda value: print(value, flush=True))
unetpp_output = Path('/kaggle/working/unetpp_rd_002_amp')
unetpp_output.mkdir(parents=True, exist_ok=True)
unetpp_previous = import_previous_run(unetpp_input_root, unetpp_output, UNETPP_CONFIG, UNETPP_REQUIRE_PREVIOUS)
for name, value in [('config.json', UNETPP_CONFIG), ('runtime.json', unetpp_runtime)]:
    path = unetpp_output/name
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Saved {name} differs; inspect before continuing the fixed run.')
    write_json_atomic(path, value)
for name, source in unetpp_sources_embedded.items():
    (unetpp_output/(name+'.py')).write_text(source)
unetpp_train.to_csv(unetpp_output/'train_manifest.csv', index=False)
unetpp_validation.to_csv(unetpp_output/'validation_manifest.csv', index=False)
write_json_atomic(unetpp_output/'input_fingerprint.json', {'fingerprint': unetpp_data_fingerprint,
    'preprocessing_identity': unetpp_preprocessing_identity, 'test_used': False})
unetpp_train_dataset = CachedClips(unetpp_paths[:len(unetpp_train)], unetpp_train['label'])
unetpp_validation_dataset = CachedClips(unetpp_paths[len(unetpp_train):], unetpp_validation['label'])
print('READY:', len(unetpp_train_dataset), 'train,', len(unetpp_validation_dataset), 'validation; no videos downloaded.')
print('Constant non-RD reference:', binary_metrics(unetpp_validation['label'], np.zeros(len(unetpp_validation))))
print('Previous run:', unetpp_previous)
''')
            add('code','kaggle-gpu-probe','''
seed_training(UNETPP_CONFIG['seed'])
unetpp_probe_indices = [int(np.flatnonzero(unetpp_train['label'].to_numpy()==label)[0]) for label in [0,1]]
unetpp_probe_inputs = torch.stack([unetpp_train_dataset[index][0] for index in unetpp_probe_indices])
unetpp_probe_labels = torch.tensor([0.,1.])
unetpp_probe = probe_training_precision(UNetPPRDClassifier, unetpp_probe_inputs, unetpp_probe_labels,
    UNETPP_CONFIG['training']['physical_batch'], UNETPP_CONFIG['training']['precision'])
write_json_atomic(unetpp_output/'precision_probe.json', unetpp_probe)
print(unetpp_probe)
del unetpp_probe_inputs, unetpp_probe_labels
gc.collect()
torch.cuda.empty_cache()
''')
            add('code','kaggle-gpu-train','''
if not isinstance(UNETPP_EPOCHS_THIS_SESSION, int) or not 1 <= UNETPP_EPOCHS_THIS_SESSION <= 50:
    raise ValueError('Choose 1–50 epochs per saved job; start with 1 for real timing.')
unetpp_session_start = time.perf_counter()
for unetpp_session_epoch in range(UNETPP_EPOCHS_THIS_SESSION):
    unetpp_summary = run_training(UNetPPRDClassifier, unetpp_train_dataset, unetpp_validation_dataset,
        UNETPP_CONFIG, unetpp_output, unetpp_data_fingerprint, unetpp_runtime,
        device='cuda', max_epochs_this_call=1, report=lambda value: print(value, flush=True))
    unetpp_history = json.loads((unetpp_output/'history.json').read_text())
    pd.DataFrame(unetpp_history).to_csv(unetpp_output/'history.csv', index=False)
    if unetpp_summary['status'] == 'completed':
        break
    elapsed = time.perf_counter()-unetpp_session_start
    last_seconds = unetpp_history[-1]['seconds_this_segment']
    if elapsed+1.5*last_seconds+180 > UNETPP_MAX_SESSION_SECONDS:
        print('Returning normally to save partial output before the session budget; continue in a new saved job.')
        break
print('Status:', unetpp_summary['status'], '| epochs:', unetpp_summary['epochs_completed'], '/ 50')
print('Best epoch by validation accuracy:', unetpp_summary['best_epoch'])
display(pd.DataFrame(unetpp_history))
''')
            add('code','kaggle-gpu-export','''
unetpp_hashes = {path.name: file_digest(path) for path in sorted(unetpp_output.iterdir())
    if path.is_file() and path.suffix in {'.json','.csv','.pt','.py'} and path.name != 'artifact_sha256.json'}
write_json_atomic(unetpp_output/'artifact_sha256.json', unetpp_hashes)
unetpp_zip = unetpp_output.parent/'unetpp_rd_002_amp_results.zip'
with zipfile.ZipFile(unetpp_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in [*unetpp_hashes, 'artifact_sha256.json']:
        archive.write(unetpp_output/name, arcname=f'{unetpp_output.name}/{name}')
print('Results ZIP:', unetpp_zip)
print('Wait for this saved Kaggle run to complete. Download its Output tab or use kaggle kernels output from VS Code.')
print('A partial job is not a completed 50-epoch baseline. No test evaluation was run.')
''')
        return {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python'},'kaggle':{'isGpuEnabled':mode=='train','isInternetEnabled':mode=='prepare'}},
            'nbformat':4,'nbformat_minor':5}
    for mode,filename in [('prepare','kaggle_unetpp_preparation.ipynb'),('train','kaggle_unetpp_training.ipynb')]:
        (ROOT/'notebooks'/filename).write_text(json.dumps(notebook(mode),indent=1)+'\n')


if __name__=='__main__':
    build()
