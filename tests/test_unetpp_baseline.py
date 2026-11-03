"""Paper-protocol behavior without ERDES data or a GPU."""
import ast
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rd_ultrasound_ml.unetpp_baseline import (
    UNetPPRDClassifier, resize_grayscale_volume, prepare_unetpp_clip,
    benchmark_cuda_batch,
)


class UNetPPBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_published_encoder_parameter_count_and_output_gradients(self):
        model = UNetPPRDClassifier()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3603553)
        self.assertEqual(len([m for m in model.modules() if isinstance(m, nn.Conv3d)]), 10)
        self.assertEqual(len([m for m in model.modules() if isinstance(m, nn.MaxPool3d)]), 4)
        self.assertFalse(any(isinstance(m, nn.BatchNorm3d) for m in model.modules()))
        logits = model(torch.rand(2, 1, 32, 32, 32))
        self.assertEqual(tuple(logits.shape), (2,))
        nn.BCEWithLogitsLoss()(logits, torch.tensor([0., 1.])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters()), 0)

    def test_temporal_resize_interpolates_instead_of_repeating_source_frames(self):
        frames = np.stack([np.full((4, 4), value, np.float32) for value in [0, 100, 200]])
        actual = resize_grayscale_volume(frames, (5, 4, 4))
        self.assertEqual(tuple(actual.shape), (1, 5, 4, 4))
        np.testing.assert_allclose(actual[0, :, 0, 0].numpy(), np.array([0, 40, 100, 160, 200]) / 255, atol=1e-6)

    def test_square_padding_and_saturated_values_remain_bounded(self):
        tensor = resize_grayscale_volume(np.full((1, 4, 8), 255, np.float32), (1, 8, 8))
        np.testing.assert_array_equal(tensor[0, 0, :2].numpy(), 0)
        np.testing.assert_array_equal(tensor[0, 0, 6:].numpy(), 0)
        np.testing.assert_array_equal(tensor[0, 0, 2:6].numpy(), 1)
        bright = resize_grayscale_volume(np.full((3, 432, 432), 255, np.float32))
        self.assertEqual(tuple(bright.shape), (1, 96, 128, 128))
        self.assertTrue(torch.isfinite(bright).all())
        self.assertLessEqual(bright.max().item(), 1)
        for invalid in [np.nan, np.inf, -1., 256.]:
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                resize_grayscale_volume(np.full((1, 4, 4), invalid, np.float32))

    def test_decoder_uses_all_frames_and_rejects_metadata_mismatch(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (48, 32))
            self.assertTrue(writer.isOpened())
            try:
                for value in [30, 100, 200]:
                    writer.write(np.full((32, 48, 3), value, np.uint8))
            finally:
                writer.release()
            result = prepare_unetpp_clip(path, 3, 32, 48, target=(5, 8, 8))
            self.assertEqual(tuple(result.shape), (1, 5, 8, 8))
            self.assertLess(result[0, 1].mean().item(), result[0, 2].mean().item())
            self.assertLess(result[0, 2].mean().item(), result[0, 3].mean().item())
            with self.assertRaisesRegex(ValueError, 'metadata'):
                prepare_unetpp_clip(path, 4)
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                prepare_unetpp_clip(path, 3, 31, 48)

    def test_instance_norm_accumulated_gradients_match_combined_batch(self):
        torch.manual_seed(42)
        # Check batch independence in float64, separating the algebra from
        # float32 convolution-kernel/reduction differences between batch sizes.
        # This does not assert bitwise equivalence for actual FP32 GPU training.
        combined, accumulated = UNetPPRDClassifier().double(), UNetPPRDClassifier().double()
        accumulated.load_state_dict(combined.state_dict())
        inputs = torch.rand(2, 1, 32, 32, 32, dtype=torch.float64)
        labels = torch.tensor([0., 1.], dtype=torch.float64)
        loss = nn.BCEWithLogitsLoss()
        loss(combined(inputs), labels).backward()
        for index in range(2):
            (loss(accumulated(inputs[index:index + 1]), labels[index:index + 1]) / 2).backward()
        for first, second in zip(combined.parameters(), accumulated.parameters()):
            torch.testing.assert_close(first.grad, second.grad, rtol=1e-6, atol=1e-9)

    def test_preflight_embeds_exact_module_and_config_and_reads_no_test_split(self):
        notebook = json.loads((ROOT / 'notebooks/unetpp_resource_check.ipynb').read_text())
        cells = {cell['id']: ''.join(cell['source']) for cell in notebook['cells']}
        self.assertEqual(cells['unetpp-implementation'], (ROOT / 'src/rd_ultrasound_ml/unetpp_baseline.py').read_text())
        embedded = ast.literal_eval(ast.parse(cells['unetpp-config']).body[0].value)
        self.assertEqual(embedded, json.loads((ROOT / 'configs/unetpp_rd_001.json').read_text()))
        opened = [n.args[0].value for n in ast.walk(ast.parse(cells['unetpp-inputs']))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and isinstance(n.func.value, ast.Name) and n.func.value.id == 'unetpp_archive'
                  and n.func.attr == 'read']
        self.assertEqual(opened, ['splits/non_rd_vs_rd/train.csv'])
        self.assertFalse(embedded['intended_training']['class_weighting'])
        self.assertEqual(embedded['intended_training']['epochs'], 50)
        self.assertEqual(embedded['intended_training']['checkpoint_selection'], 'highest validation accuracy; earliest on ties')

    def test_cpu_session_cannot_be_misreported_as_a_gpu_benchmark(self):
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                benchmark_cuda_batch(UNetPPRDClassifier, torch.zeros(2, 1, 32, 32, 32), torch.zeros(2), 1)


if __name__ == '__main__':
    unittest.main()
