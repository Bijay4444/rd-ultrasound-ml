"""Analytical attribution tests and control behavior on small volumes."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from torch import nn
from rd_ultrasound_ml.video_explainability import (
    ModelAdapter, assert_state, gradcam, occlusion, randomized_maps, select_cases, snapshot,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv3d(1, 1, 1, bias=False)
        self.head = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.features.weight.fill_(1); self.head.weight.fill_(2)

    def forward(self, x):
        return self.head(self.features(x).mean((2, 3, 4))).flatten()


class ExplainabilityTests(unittest.TestCase):
    def setUp(self):
        self.model = Tiny().eval()
        self.x = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4)/63

    def test_analytical_cam_and_unchanged_state(self):
        before = snapshot(self.model)
        result = gradcam(self.model, self.x, 'features')
        np.testing.assert_allclose(result['raw'], self.x.numpy()[0, 0]*2/64, atol=1e-8)
        np.testing.assert_allclose(result['normalized'], self.x.numpy()[0, 0], atol=1e-6)
        assert_state(self.model, before)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertEqual(len(self.model.features._forward_hooks), 0)

    def test_negative_target_and_constant_map(self):
        self.assertTrue(gradcam(self.model, self.x, 'features', -1)['zero_map'])
        result = gradcam(self.model, torch.ones_like(self.x), 'features')
        self.assertTrue(result['constant_map']); self.assertFalse(result['zero_map'])
        self.assertEqual(result['normalized'].max(), 0)

    def test_requires_eval_and_valid_inputs(self):
        with self.assertRaises(ValueError): gradcam(self.model.train(), self.x, 'features')
        self.model.eval()
        with self.assertRaises(ValueError): gradcam(self.model, self.x.double(), 'features')
        self.x[0, 0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError): gradcam(self.model, self.x, 'features')

    def test_hook_cleanup_on_bad_layer_output(self):
        with self.assertRaises(ValueError): gradcam(self.model, self.x, 'head')
        self.assertFalse(self.model.head._forward_hooks)

    def test_frozen_parameters_and_disabled_outer_grad(self):
        self.model.requires_grad_(False)
        with torch.no_grad(): result = gradcam(self.model, self.x, 'features')
        self.assertGreater(result['raw'].max(), 0)

    def test_occlusion_sensitivity_and_replay(self):
        attribution = self.x.numpy()[0, 0]
        first = occlusion(self.model, self.x, attribution, cuboid=(2, 2, 2), count=2)
        self.assertEqual(first, occlusion(self.model, self.x, attribution, cuboid=(2, 2, 2), count=2))
        self.assertGreater(first['groups']['high']['mean_drop'], first['groups']['low']['mean_drop'])

    def test_randomization_replay_state_and_rng(self):
        adapter = ModelAdapter('tiny', Tiny, 'features', 'head', {})
        reference = gradcam(self.model, self.x, 'features')['normalized']
        before = snapshot(self.model); rng = torch.random.get_rng_state().clone()
        first = list(randomized_maps(self.model, self.x, adapter, reference))
        second = list(randomized_maps(self.model, self.x, adapter, reference))
        self.assertEqual(len(first), 3)
        for a, b in zip(first, second): np.testing.assert_array_equal(a['raw'], b['raw'])
        assert_state(self.model, before)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_selection_by_id_with_nondefault_index(self):
        frame = pd.DataFrame({'clip_id': list('hgfedcba'), 'label': [1, 1, 0, 0, 0, 0, 1, 1]}, index=range(10, 18))
        probabilities = [.9, .8, .1, .2, .8, .9, .1, .2]
        cases = select_cases(frame, probabilities)
        self.assertEqual([c['clip_id'] for c in cases], list('ghefcdab'))
        with self.assertRaises(ValueError): select_cases(frame.iloc[:3], probabilities[:3])

    def test_real_unet_forward(self):
        from rd_ultrasound_ml.unetpp_baseline import UNetPPRDClassifier
        torch.set_num_threads(2)
        inputs = torch.rand(1, 1, 32, 32, 32)
        model = UNetPPRDClassifier().eval()
        result = gradcam(model, inputs, 'encoder.4')
        self.assertEqual(result['raw'].shape, (32, 32, 32))
        self.assertEqual(result['activation_shape'], [1, 256, 2, 2, 2])

    def test_real_r2_forward(self):
        try:
            from rd_ultrasound_ml.r2plus1d18_baseline import R2Plus1D18RDClassifier
        except ModuleNotFoundError as error:
            if error.name == 'torchvision': self.skipTest('torchvision unavailable in this environment')
            raise
        torch.set_num_threads(2)
        model = R2Plus1D18RDClassifier().eval()
        before = snapshot(model)
        result = gradcam(model, torch.rand(1, 1, 8, 32, 32), 'backbone.layer4')
        self.assertEqual(result['activation_shape'], [1, 512, 1, 2, 2])
        assert_state(model, before)


if __name__ == '__main__': unittest.main()
