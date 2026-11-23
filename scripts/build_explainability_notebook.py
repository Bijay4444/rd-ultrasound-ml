"""Build standalone Kaggle notebook from frozen sources and explanation modules."""
import json
from pathlib import Path
import pprint

ROOT = Path(__file__).resolve().parents[1]
MODULES = ['unetpp_baseline', 'unetpp_training', 'unetpp_data', 'unetpp_kaggle',
           'video_cache', 'comparison_export', 'r2plus1d18_baseline', 'unetpp_evaluation',
           'comparison_evaluation', 'video_explainability', 'explainability_workflow']


def notebook():
    training = json.loads((ROOT/'notebooks/kaggle_r2plus1d18_training.ipynb').read_text())
    shared = {c['id']: ''.join(c['source']) for c in training['cells']}
    cells = []

    def add(kind, ident, source):
        cell = {'cell_type': kind, 'id': ident, 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code': cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', 'intro', '''# Frozen-model validation explanations

Attach exactly these saved Kaggle outputs as inputs:
- `bijaykhadka1710/unetpp-rd-002-inputs-00` version 2
- `bijaykhadka1710/unetpp-rd-002-inputs-01` version 2
- `bijaykhadka1710/unetpp-rd-002-train-03` version 1
- `namanbabariya/r2plus1d18-rd-001-train-04` version 1

Select one T4 GPU and save a version with all cells executed. No Internet or
test-input output is required. The complete cache is verified before selecting
eight fixed validation clips. Source and checkpoint hashes are enforced.
FP32 attribution uses the original tensor shape. An OOM or parity failure stops
the run and exports its failure status; there is no silent fallback.

Outputs: `/kaggle/working/gradcam_validation_001.zip`. If rerunning, use a fresh
working directory rather than mixing artifacts. Heatmaps attribute logits;
they are not segmentation masks or evidence of clinical validity.
''')
    add('code', 'runtime', shared['r2p1-runtime'])
    add('code', 'module-loader', shared['r2p1-module-loader'])
    for name in MODULES:
        source = (ROOT/f'src/rd_ultrasound_ml/{name}.py').read_text()
        add('code', 'embed-'+name, 'r2p1_load_module('+repr(name)+', '+repr(source)+')\n')
    freezes = {name: json.loads((ROOT/f'configs/{filename}').read_text()) for name, filename in (
        ('unetpp', 'unetpp_rd_002_evaluation.json'), ('r2plus1d18', 'r2plus1d18_rd_001_evaluation.json'))}
    add('code', 'frozen-models', 'FREEZES = '+pprint.pformat(freezes, width=100)+'\n'+'''
from rd_ultrasound_ml.unetpp_baseline import UNetPPRDClassifier
from rd_ultrasound_ml.r2plus1d18_baseline import R2Plus1D18RDClassifier
from rd_ultrasound_ml.video_explainability import ModelAdapter
from rd_ultrasound_ml.explainability_workflow import run_explanations
adapters = [ModelAdapter('unetpp', UNetPPRDClassifier, 'encoder.4', 'head', FREEZES['unetpp']),
            ModelAdapter('r2plus1d18', R2Plus1D18RDClassifier, 'backbone.layer4', 'backbone.fc', FREEZES['r2plus1d18'])]
''')
    add('code', 'explain', '''result = run_explanations(Path('/kaggle/input'),
    Path('/kaggle/working/gradcam_validation_001'), adapters, r2p1_sources_embedded, runtime=r2p1_runtime)
print(result)
''')
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}, 'kaggle': {'isGpuEnabled': True, 'isInternetEnabled': False}},
            'nbformat': 4, 'nbformat_minor': 5}


if __name__ == '__main__':
    target = ROOT/'notebooks/kaggle_video_explainability.ipynb'
    target.write_text(json.dumps(notebook(), indent=1)+'\n')
    print(target)
