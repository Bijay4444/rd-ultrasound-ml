"""Build the frozen R(2+1)D-18 reused-official-test evaluation notebook.

Reads configs/r2plus1d18_rd_001_evaluation.json, the tracked modules and the shared
runtime/module-loader cells of the R(2+1)D-18 training notebook; writes only
notebooks/kaggle_r2plus1d18_evaluation.ipynb. U-Net++ files are untouched.
"""
import json
from pathlib import Path
import pprint
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT/'configs/r2plus1d18_rd_001_evaluation.json'
TRAINING_NOTEBOOK = ROOT/'notebooks/kaggle_r2plus1d18_training.ipynb'
NOTEBOOK = ROOT/'notebooks/kaggle_r2plus1d18_evaluation.ipynb'
TRAINING_MODULES = ['unetpp_baseline', 'unetpp_training', 'unetpp_data', 'unetpp_kaggle',
                    'video_cache', 'comparison_export', 'r2plus1d18_baseline']
MODULES = [*TRAINING_MODULES, 'unetpp_evaluation', 'comparison_evaluation']


def notebook():
    freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
    training = json.loads(TRAINING_NOTEBOOK.read_text(encoding='utf-8'))
    shared = {cell['id']: ''.join(cell['source']) for cell in training['cells']}
    cells = []

    def add(kind, ident, source):
        cell = {'cell_type': kind, 'id': ident, 'metadata': {},
                'source': dedent(source).lstrip('\n').splitlines(keepends=True)}
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n') + '\n'
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', 'r2p1-eval-intro', '''
# R(2+1)D-18 RD: frozen evaluation on the reused official test split

**Do not run until the team has frozen all model choices.** The embedded freeze
(`configs/r2plus1d18_rd_001_evaluation.json`) must have `team_freeze_confirmed: true`
from a reviewed commit; otherwise this notebook stops before opening any test data.

Attach (read-only) the final training output `r2plus1d18-rd-001-train-04` and the
saved U-Net++ test-input output `unetpp-rd-002-test-inputs`. The notebook verifies
the training artifact (config digest, run fingerprint, 50/50 epochs, selected
**epoch 15**, checkpoint SHA256) *before* reading `test.csv`, checks the GPU runtime
equals training, re-verifies all 1,077 cached test tensors, then runs one forward
pass per clip at **threshold 0.5**. No optimizer, retraining, threshold search,
calibration or epoch selection. Inference timing is recorded.

The official test split was already used for the U-Net++ result, so report this as
a **reused official test benchmark**. Clip-level splits may share patients; this
educational result is not a clinical or unseen-patient validation.
''')
    add('code', 'r2p1-eval-runtime', shared['r2p1-runtime'])
    add('code', 'r2p1-eval-freeze', 'R2P1_FREEZE = ' + pprint.pformat(freeze, sort_dicts=False, width=100) + '\n')
    add('code', 'r2p1-eval-module-loader', shared['r2p1-module-loader'])
    for name in MODULES:
        source = (ROOT/f'src/rd_ultrasound_ml/{name}.py').read_text(encoding='utf-8')
        add('code', 'r2p1-eval-embed-' + name,
            'R2P1_MODULE_SOURCE = ' + repr(source) + '\nr2p1_load_module(' + repr(name) + ', R2P1_MODULE_SOURCE)\n')
    add('code', 'r2p1-eval-verify-training', '''
from rd_ultrasound_ml.r2plus1d18_baseline import R2Plus1D18RDClassifier
from rd_ultrasound_ml.unetpp_training import file_digest, json_digest, write_json_atomic
from rd_ultrasound_ml.comparison_evaluation import (find_comparison_training, load_reused_test_cache,
    evaluate_comparison_model)
if R2P1_FREEZE.get('team_freeze_confirmed') is not True:
    raise RuntimeError('Team freeze not confirmed in the evaluation config; no test data has been opened.')
for name, digest in R2P1_FREEZE['training_implementation_sha256'].items():
    if hashlib.sha256(r2p1_sources_embedded[name].encode()).hexdigest() != digest:
        raise ValueError('Embedded training implementation differs from freeze: ' + name)
r2p1_training_dir = find_comparison_training(Path('/kaggle/input'), R2P1_FREEZE)
print('Frozen training verified: epoch', R2P1_FREEZE['selected_epoch'], '| checkpoint SHA256:',
      R2P1_FREEZE['checkpoint_sha256'])
''')
    add('code', 'r2p1-eval-gpu-inputs', '''
if not torch.cuda.is_available():
    raise RuntimeError('Select an available Kaggle CUDA GPU for frozen test inference.')
if r2p1_runtime != R2P1_FREEZE['training_runtime'] or json.loads((r2p1_training_dir/'runtime.json').read_text()) != r2p1_runtime:
    raise ValueError('GPU runtime differs from frozen training; no test data opened. Inspect and document before inference.')
r2p1_test, r2p1_test_dataset, r2p1_test_record = load_reused_test_cache(
    Path('/kaggle/input'), r2p1_training_dir, R2P1_FREEZE, report=lambda value: print(value, flush=True))
r2p1_output = Path('/kaggle/working') / R2P1_FREEZE['evaluation_id']
r2p1_output.mkdir(parents=True, exist_ok=True)
write_json_atomic(r2p1_output/'runtime.json', r2p1_runtime)
write_json_atomic(r2p1_output/'test_input_provenance.json', r2p1_test_record)
for name, source in r2p1_sources_embedded.items():
    (r2p1_output/(name+'.py')).write_text(source)
print('READY for frozen inference:', len(r2p1_test_dataset), 'official test clips.')
''')
    add('code', 'r2p1-eval-inference', '''
r2p1_result = evaluate_comparison_model(R2Plus1D18RDClassifier, r2p1_test_dataset, r2p1_test,
    r2p1_training_dir, R2P1_FREEZE, r2p1_output, r2p1_runtime,
    device='cuda', report=lambda value: print(value, flush=True))
print('REUSED OFFICIAL TEST METRICS:', r2p1_result['metrics'])
print('Inference timing:', r2p1_result['inference_timing'])
display(pd.read_csv(r2p1_output/'test_reference_comparison.csv'))
''')
    add('code', 'r2p1-eval-export', '''
r2p1_hashes = {path.name: file_digest(path) for path in sorted(r2p1_output.iterdir())
    if path.is_file() and path.suffix in {'.json', '.csv', '.py'} and path.name != 'artifact_sha256.json'}
write_json_atomic(r2p1_output/'artifact_sha256.json', r2p1_hashes)
r2p1_zip = r2p1_output.parent/(R2P1_FREEZE['evaluation_id'] + '_results.zip')
with zipfile.ZipFile(r2p1_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in [*r2p1_hashes, 'artifact_sha256.json']:
        archive.write(r2p1_output/name, arcname=f'{r2p1_output.name}/{name}')
print('Saved test results ZIP:', r2p1_zip)
print('Single frozen evaluation complete. No model was trained or tuned in this notebook.')
''')
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'kaggle': {'isGpuEnabled': True, 'isInternetEnabled': False}},
        'nbformat': 4, 'nbformat_minor': 5}


def build():
    NOTEBOOK.write_text(json.dumps(notebook(), indent=1) + '\n', encoding='utf-8')
    return NOTEBOOK


if __name__ == '__main__':
    print(build())
