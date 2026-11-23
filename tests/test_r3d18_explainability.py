"""R3D extension identity, subset orchestration and analytical delivery checks."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'),str(ROOT/'src')]
from build_r3d18_explainability_notebook import notebook, NOTEBOOK, CONFIG
from prepare_r3d18_explainability_job import prepare, SOURCES
from rd_ultrasound_ml.explainability_workflow import run_explanations
from rd_ultrasound_ml.video_explainability import ModelAdapter, gradcam, snapshot, assert_state, select_cases


class R3DExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_notebook_freezes_sources_and_original_case_manifest(self):
        result=notebook()
        self.assertEqual(json.loads(NOTEBOOK.read_text()),result)
        nbformat.validate(nbformat.from_dict(result))
        sources={}
        cells={c['id']:''.join(c['source']) for c in result['cells']}
        for c in result['cells']:
            if c['cell_type']=='code':
                compile(''.join(c['source']),c['id'],'exec')
                self.assertEqual(c['outputs'],[])
                if c['id'].startswith('embed-'):
                    name,source=map(ast.literal_eval,ast.parse(''.join(c['source'])).body[0].value.args)
                    self.assertEqual(source,(ROOT/f'src/rd_ultrasound_ml/{name}.py').read_text())
                    sources[name]=source
        tree=ast.parse(cells['frozen-models'])
        freezes,plan=[ast.literal_eval(s.value) for s in tree.body[:2]]
        self.assertEqual(plan,json.loads(CONFIG.read_text()))
        self.assertEqual([r['clip_id'] for r in plan['selection']['cases']],
            ['102769_00005','102769_00009','164267_00003','164267_00005',
             '164267_00944','164267_01736','405744_00014','755384_00001'])
        self.assertEqual(plan['explain_models'],['r3d18'])
        for name,freeze in freezes.items():
            for module,expected in freeze['training_implementation_sha256'].items():
                self.assertEqual(hashlib.sha256(sources[module].encode()).hexdigest(),expected)
        self.assertEqual(freezes['r3d18']['selected_epoch'],18)
        self.assertIn("explain_models=EXPLANATION_PLAN['explain_models']",cells['explain'])

    def test_private_pinned_bundle_excludes_test_inputs(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(ValueError):prepare('example',d,'unpinned')
            p=prepare('example',d,'registry/image@sha256:'+'a'*64)
            self.assertEqual(p,prepare('example',d,'registry/image@sha256:'+'a'*64))
            meta=json.loads((p/'kernel-metadata.json').read_text())
            self.assertTrue(meta['is_private'])
            self.assertFalse(meta['enable_internet'])
            self.assertEqual(meta['kernel_sources'],SOURCES)
            self.assertFalse(any('test' in s for s in SOURCES))

    def test_real_r3d_attribution_preserves_state_and_hooks(self):
        from rd_ultrasound_ml.r3d18_baseline import R3D18RDClassifier
        model=R3D18RDClassifier().eval()
        state=snapshot(model)
        cam=gradcam(model,torch.rand(1,1,8,32,32),'backbone.layer4')
        self.assertEqual(cam['activation_shape'],[1,512,1,2,2])
        self.assertTrue(np.isfinite(cam['raw']).all())
        self.assertEqual(cam['raw'].shape,(8,32,32))
        assert_state(model,state)
        self.assertFalse(model.backbone.layer4._forward_hooks)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_invalid_subset_fails_before_cache_or_model_access(self):
        adapters=[ModelAdapter('unetpp',None,'features','head',{}),
                  ModelAdapter('r3d18',None,'features','head',{})]
        with TemporaryDirectory() as d:
            for i,names in enumerate([[],['unknown'],['r3d18','r3d18']]):
                with patch('rd_ultrasound_ml.explainability_workflow.read_preparation_config') as read:
                    with self.assertRaisesRegex(ValueError,'subset'):
                        run_explanations(d,Path(d)/str(i),adapters,{},device='cpu',explain_models=names)
                    read.assert_not_called()

    def test_case_drift_fails_before_explanation_inference(self):
        import rd_ultrasound_ml.explainability_workflow as workflow
        frame=pd.DataFrame({'clip_id':list('abcdefgh'),'label':[1,1,0,0,0,0,1,1]})
        scores=np.array([.8,.9,.1,.2,.8,.9,.1,.2])
        cases=select_cases(frame,scores)
        adapters=[ModelAdapter('unetpp',None,'features','head',{
            'training_implementation_sha256':{},'training_data_fingerprint':'cache'})]
        with TemporaryDirectory() as d:
            (Path(d)/'config.json').write_text('{}')
            with patch.object(workflow,'read_preparation_config',return_value=({},None)), \
                 patch.object(workflow,'load_prepared_shards',return_value=(frame,frame,[],'cache',{})), \
                 patch.object(workflow,'find_frozen_training',return_value=Path(d)), \
                 patch.object(workflow,'prediction_record',return_value=scores), \
                 patch.object(workflow.torch,'load') as load:
                wrong=[{**cases[0],'clip_id':'changed'},*cases[1:]]
                with self.assertRaisesRegex(ValueError,'historical panel'):
                    run_explanations(d,Path(d)/'output',adapters,{},device='cpu',expected_cases=wrong)
                load.assert_not_called()


if __name__=='__main__':unittest.main()
