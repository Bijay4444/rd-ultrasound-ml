"""Preprocessing behavior using synthetic frames and a short local video."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np


class BaselinePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'notebooks/baseline_preparation.ipynb'
        notebook = json.loads(path.read_text())
        cell = next(c for c in notebook['cells'] if c['id'] == 'prep-functions')
        cls.functions = {'np': np, 'cv2': cv2, 'Path': Path}
        exec(compile(ast.parse(''.join(cell['source'])), 'prep-functions', 'exec'), cls.functions)

    def test_short_clips_repeat_ordered_indices_and_include_endpoints(self):
        indices = self.functions['uniform_frame_indices'](3, 5)
        np.testing.assert_array_equal(indices, [0, 0, 1, 2, 2])
        for count, target in [(1, 16), (90, 16), (3, 16)]:
            indices = self.functions['uniform_frame_indices'](count, target)
            self.assertEqual(len(indices), target)
            self.assertEqual(indices[0], 0)
            self.assertEqual(indices[-1], count - 1)
            self.assertTrue(np.all(np.diff(indices) >= 0))
        with self.assertRaises(ValueError):
            self.functions['uniform_frame_indices'](0, 16)

    def test_frame_padding_grayscale_and_normalization(self):
        frame = np.zeros((4, 8, 3), dtype=np.uint8)
        frame[:, :, 0] = 255
        result = self.functions['prepare_frame'](frame, 8)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (8, 8))
        np.testing.assert_array_equal(result[:2], 0)
        np.testing.assert_array_equal(result[6:], 0)
        np.testing.assert_allclose(result[2:6], 1 / 3, atol=1e-6)

    def test_video_layout_repeated_frames_and_metadata_mismatch(self):
        with TemporaryDirectory(prefix='erdes-preprocessing-') as tmp:
            path = Path(tmp) / 'three_frames.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (48, 32))
            self.assertTrue(writer.isOpened(), 'MJPG fixture encoder unavailable')
            try:
                for value in [30, 100, 200]:
                    writer.write(np.full((32, 48, 3), value, dtype=np.uint8))
            finally:
                writer.release()
            clip, indices = self.functions['prepare_clip'](path, 3, 5, 8)
            self.assertEqual(clip.shape, (1, 5, 8, 8))
            self.assertEqual(clip.dtype, np.float32)
            self.assertGreaterEqual(clip.min(), 0)
            self.assertLessEqual(clip.max(), 1)
            np.testing.assert_array_equal(clip[:, 0], clip[:, 1])
            np.testing.assert_array_equal(clip[:, 3], clip[:, 4])
            self.assertLess(clip[:, 0].mean(), clip[:, 4].mean())
            with self.assertRaisesRegex(ValueError, 'decoded 3 frames; metadata expects 4'):
                self.functions['prepare_clip'](path, 4, 5, 8)

    def test_missing_video_fails_clearly(self):
        with TemporaryDirectory(prefix='erdes-missing-video-') as tmp:
            with self.assertRaises(FileNotFoundError):
                self.functions['prepare_clip'](Path(tmp) / 'missing.mp4', 3)


if __name__ == '__main__':
    unittest.main()
