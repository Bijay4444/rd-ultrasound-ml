"""Video transform semantics, input immutability and interrupted training replay."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rd_ultrasound_ml.unetpp_data import CachedClips
from rd_ultrasound_ml.unetpp_training import CheckpointStore
from rd_ultrasound_ml.video_augmentation import (
    CANDIDATE_POLICY, TrainingAugmentedClips, apply_parameters, augment_clip,
    run_augmented_training, sample_parameters, validate_policy,
)


def always_policy():
    return {**deepcopy(CANDIDATE_POLICY), 'affine_probability': 1.0, 'gain_probability': 1.0}


def parameters(**updates):
    return {'affine_applied': True, 'gain_applied': False, 'rotation_degrees': 0.0,
        'translation_x_fraction': 0.0, 'translation_y_fraction': 0.0,
        'scale': 1.0, 'gain': 1.0, **updates}


def tiny_model():
    return nn.Sequential(nn.Conv3d(1, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool3d(1),
        nn.Flatten(), nn.Dropout(.25), nn.Linear(2, 1))


class VideoAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_returns_independent_copy(self):
        x = torch.rand(1, 5, 12, 16)
        before = x.clone()
        policy = {**deepcopy(CANDIDATE_POLICY), 'affine_probability': 0.0, 'gain_probability': 0.0}
        y, chosen = augment_clip(x, policy, 42, 1, 'clip')
        self.assertTrue(torch.equal(x, y))
        self.assertFalse(chosen['affine_applied'])
        y.zero_()
        self.assertTrue(torch.equal(x, before))

    def test_translation_matches_pixel_space_oracle(self):
        x = torch.zeros(1, 3, 8, 10)
        x[:, :, 2:5, 3:6] = .75
        y = apply_parameters(x, parameters(translation_x_fraction=.1, translation_y_fraction=.25))
        expected = torch.zeros_like(x)
        expected[:, :, 2:, 1:] = x[:, :, :-2, :-1]
        torch.testing.assert_close(y, expected, rtol=0, atol=1e-6)

    def test_rectangular_rotation_uses_pixel_aspect_ratio(self):
        x = torch.zeros(1, 2, 13, 21)
        x[:, :, 6, 12] = 1
        y = apply_parameters(x, parameters(rotation_degrees=90))
        self.assertAlmostEqual(y[0, 0, 8, 10].item(), 1, places=5)
        self.assertAlmostEqual(y.sum().item(), 2, places=5)

    def test_temporal_order_and_shared_parameters(self):
        base = torch.rand(1, 1, 16, 16) * .5
        x = torch.cat([base * a for a in [.25, .5, .75, 1]], dim=1)
        y, _ = augment_clip(x, always_policy(), 42, 1, 'temporal')
        self.assertEqual(x.shape, y.shape)
        for i, factor in enumerate([.25, .5, .75, 1]):
            torch.testing.assert_close(y[:, i], y[:, -1] * factor, rtol=1e-6, atol=1e-7)

    def test_gain_and_clamping(self):
        x = torch.tensor([[[[.2, .8, 1.0]]]], dtype=torch.float32)
        y = apply_parameters(x, parameters(affine_applied=False, gain_applied=True, gain=1.1))
        torch.testing.assert_close(y, torch.tensor([[[[.22, .88, 1.0]]]]))
        self.assertTrue(torch.equal(x, torch.tensor([[[[.2, .8, 1.0]]]])))

    def test_keyed_views_replay_without_consuming_global_rng(self):
        before = torch.get_rng_state().clone()
        first = sample_parameters(always_policy(), 42, 1, 'one')
        sample_parameters(always_policy(), 42, 1, 'two')
        self.assertEqual(first, sample_parameters(always_policy(), 42, 1, 'one'))
        for seed, epoch, clip in [(43, 1, 'one'), (42, 2, 'one'), (42, 1, 'two')]:
            self.assertNotEqual(first, sample_parameters(always_policy(), seed, epoch, clip))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_invalid_policy_inputs_and_partition_are_rejected(self):
        for key, value in [('affine_probability', 2), ('rotation_degrees', float('nan')),
                           ('gain_range', [-1, 1]), ('order', ['gain', 'affine']), ('padding_mode', 'border')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_policy({**deepcopy(CANDIDATE_POLICY), key: value})
        for x in [torch.zeros(2, 4, 8, 8), torch.zeros(1, 4, 8, 8).double(),
                  torch.full((1, 4, 8, 8), float('nan')), torch.ones(1, 4, 8, 8) * 2]:
            with self.assertRaises(ValueError):
                augment_clip(x, CANDIDATE_POLICY, 42, 1, 'one')
        with self.assertRaises(ValueError):
            TrainingAugmentedClips([None], ['one'], CANDIDATE_POLICY, 42, partition='validation')

    def test_cached_arrays_and_labels_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'clip.npy'
            np.save(path, np.random.default_rng(42).random((1, 4, 12, 12), dtype=np.float32))
            original = path.read_bytes()
            clean = CachedClips([path], [1])
            wrapped = TrainingAugmentedClips(clean, ['one'], always_policy(), 42)
            first, label = wrapped[0]
            wrapped.set_epoch(2)
            second, second_label = wrapped[0]
            self.assertFalse(torch.equal(first, second))
            self.assertEqual(label.item(), second_label.item())
            self.assertEqual(label.item(), 1)
            self.assertEqual(path.read_bytes(), original)
            torch.testing.assert_close(clean[0][0], torch.from_numpy(np.load(path)), rtol=0, atol=0)
            self.assertTrue(torch.isfinite(first).all() and torch.isfinite(second).all())
            self.assertTrue(0 <= first.min() <= first.max() <= 1)

    def test_interruption_and_session_resume_match_uninterrupted_training(self):
        inputs = torch.rand(7, 1, 4, 12, 12, generator=torch.Generator().manual_seed(100))
        labels = torch.tensor([0., 1., 0., 1., 0., 1., 0.])
        clean = TensorDataset(inputs, labels)
        original_inputs = inputs.clone()
        config = {'experiment_id': 'synthetic_augmentation', 'seed': 42,
            'training_augmentation': always_policy(),
            'training': {'epochs': 3, 'physical_batch': 2, 'effective_batch': 4,
                'precision': 'float32', 'learning_rate': .01, 'betas': [.9, .999],
                'eps': 1e-8, 'weight_decay': 0., 'checkpoint_every_updates': 1},
            'evaluation': {'threshold': .5}}
        def wrapped():
            return TrainingAugmentedClips(clean, [str(i) for i in range(7)], always_policy(), 42)
        def run(dataset, root):
            return run_augmented_training(tiny_model, dataset, clean, config, root,
                'fixed', {}, device='cpu', report=lambda _: None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for _ in range(3):
                direct = run(wrapped(), root / 'direct')
            original_save = CheckpointStore.save
            interrupted = False
            def save_then_interrupt(store, payload):
                nonlocal interrupted
                original_save(store, payload)
                if payload['phase'] == 'train' and payload['progress']['position'] == 4 and not interrupted:
                    interrupted = True
                    raise ConnectionError('synthetic interruption after augmented optimizer update')
            with patch.object(CheckpointStore, 'save', save_then_interrupt), self.assertRaises(ConnectionError):
                run(wrapped(), root / 'resumed')
            for _ in range(3):
                resumed = run(wrapped(), root / 'resumed')
            self.assertEqual(resumed['status'], 'completed')
            self.assertEqual(resumed['best_epoch'], direct['best_epoch'])
            a, b = CheckpointStore(root / 'direct').load(), CheckpointStore(root / 'resumed').load()
            for key in a['model_state']:
                torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)
            for key in ('scheduler_state', 'scaler_state', 'best_epoch', 'best_accuracy', 'order', 'progress'):
                self.assertEqual(a[key], b[key])
            for left, right in zip(a['history'], b['history']):
                for key in left.keys() - {'seconds_this_segment', 'validation_seconds', 'resumed_from_position'}:
                    self.assertEqual(left[key], right[key])
            for key in a['optimizer_state']['state']:
                for field, value in a['optimizer_state']['state'][key].items():
                    torch.testing.assert_close(value, b['optimizer_state']['state'][key][field], rtol=0, atol=0)
            changed = deepcopy(config)
            changed['training_augmentation']['rotation_degrees'] = 3
            altered = TrainingAugmentedClips(clean, [str(i) for i in range(7)], changed['training_augmentation'], 42)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                run_augmented_training(tiny_model, altered, clean, changed, root / 'resumed', 'fixed', {}, device='cpu')
        self.assertTrue(torch.equal(inputs, original_inputs))

    def test_frozen_baseline_sources_remain_exact(self):
        freeze = json.loads((ROOT / 'configs/unetpp_rd_002_evaluation.json').read_text())
        for name, digest in freeze['training_implementation_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / f'src/rd_ultrasound_ml/{name}.py').read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
