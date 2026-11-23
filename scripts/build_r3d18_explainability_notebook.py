"""Build a validation-only R3D explanation extension for the historical eight cases."""
import json
from pathlib import Path
import pprint

from build_explainability_notebook import notebook as reference_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/'notebooks/kaggle_r3d18_explainability.ipynb'
CONFIG = ROOT/'configs/r3d18_gradcam_validation_001.json'


def notebook():
    result = reference_notebook()
    cells = result['cells']
    def replace(ident, source):
        next(c for c in cells if c['id'] == ident)['source'] = source.splitlines(keepends=True)
    replace('intro', '''# R3D-18: extend the fixed validation explanation panel

Explains only frozen R3D-18 epoch 18; U-Net++ epoch 48 supplies the original
case-selection reference. The same eight cases must match the historical panel
before explanation inference. No training or official-test data access occurs.
Original and non-RD evidence, parity, state preservation, occlusion and cumulative
randomization controls use the existing method. No precision/resolution fallback.

Attach version-pinned training/validation caches and training results:
- bijaykhadka1710/unetpp-rd-002-inputs-00/2
- bijaykhadka1710/unetpp-rd-002-inputs-01/2
- bijaykhadka1710/unetpp-rd-002-train-03/1
- namanbabariya/r3d18-rd-001-train-03/1

Output: /kaggle/working/gradcam_r3d18_validation_001.zip.
Combine verified exports with the historical two-model archive locally. Maps
attribute a model score, not anatomical ground truth or clinical validity.
''')
    cell = next(c for c in cells if c['id'] == 'embed-r2plus1d18_baseline')
    cell['id'] = 'embed-r3d18_baseline'
    cell['source'] = ['r2p1_load_module('+repr('r3d18_baseline')+', '+
                      repr((ROOT/'src/rd_ultrasound_ml/r3d18_baseline.py').read_text())+')\n']
    freezes = {name: json.loads((ROOT/'configs'/filename).read_text()) for name, filename in (
        ('unetpp','unetpp_rd_002_evaluation.json'), ('r3d18','r3d18_rd_001_evaluation.json'))}
    plan = json.loads(CONFIG.read_text())
    replace('frozen-models', 'FREEZES = '+pprint.pformat(freezes, width=100)+'\n'+
        'EXPLANATION_PLAN = '+pprint.pformat(plan, width=100)+'\n'+'''
from rd_ultrasound_ml.unetpp_baseline import UNetPPRDClassifier
from rd_ultrasound_ml.r3d18_baseline import R3D18RDClassifier
from rd_ultrasound_ml.video_explainability import ModelAdapter
from rd_ultrasound_ml.explainability_workflow import run_explanations
adapters = [ModelAdapter('unetpp', UNetPPRDClassifier, 'encoder.4', 'head', FREEZES['unetpp']),
            ModelAdapter('r3d18', R3D18RDClassifier, 'backbone.layer4', 'backbone.fc', FREEZES['r3d18'])]
''')
    replace('explain', '''result = run_explanations(Path('/kaggle/input'),
    Path('/kaggle/working') / EXPLANATION_PLAN['run_id'], adapters, r2p1_sources_embedded,
    runtime=r2p1_runtime, explain_models=EXPLANATION_PLAN['explain_models'],
    expected_cases=EXPLANATION_PLAN['selection']['cases'])
print(result)
''')
    return result


if __name__ == '__main__':
    NOTEBOOK.write_text(json.dumps(notebook(), indent=1)+'\n')
    print(NOTEBOOK)
