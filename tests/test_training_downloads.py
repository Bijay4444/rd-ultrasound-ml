"""Replay interrupted pilot acquisition without network, data, or ML packages."""
import ast
from contextlib import redirect_stdout
import hashlib
from http.client import IncompleteRead
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[1]


class TrainingDownloadTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads((ROOT / 'notebooks/baseline_training.ipynb').read_text())
        cells = {cell['id']: ''.join(cell['source']) for cell in notebook['cells']}
        self.scope = {'Path': Path, 'hashlib': hashlib}
        helper = cells.get('pilot-download-helper', cells['pilot-runtime'])
        for source in [helper, cells['pilot-download-functions']]:
            tree = ast.parse(source)
            definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                           or (source == cells.get('pilot-download-helper') and isinstance(node, (ast.Import, ast.ImportFrom)))]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), 'download-definitions', 'exec'), self.scope)
        self.scope.update(HTTPError=HTTPError, URLError=URLError, IncompleteRead=IncompleteRead,
                          time=SimpleNamespace(sleep=Mock()), urlopen=Mock())

    def test_503_after_100_cached_clips_recovers_and_rerun_uses_cache(self):
        valid = b'verified video bytes'
        digest = hashlib.sha256(valid).hexdigest()
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            clips = data_dir / 'clips'
            clips.mkdir()
            rows = [SimpleNamespace(file_path=f'clip_{i:03d}.mp4') for i in range(101)]
            for row in rows[:100]:
                (clips / row.file_path).write_bytes(valid)
            class Frame:
                def __len__(self): return len(rows)
                def __getitem__(self, key): return [row.file_path for row in rows]
                def itertuples(self, index=False): return iter(rows)
            self.scope.update(data_dir=data_dir, sources={row.file_path: {
                'url': 'https://example.invalid/' + row.file_path,
                'sha256': digest, 'bytes': len(valid)} for row in rows})
            self.scope['urlopen'].side_effect = [
                HTTPError('https://example.invalid/clip_100.mp4', 503, 'Service Unavailable', {}, None),
                io.BytesIO(valid),
            ]
            with redirect_stdout(io.StringIO()):
                self.scope['acquire_videos'](Frame())
                self.scope['acquire_videos'](Frame())
            self.assertEqual(self.scope['urlopen'].call_count, 2)
            self.assertEqual(len(list(clips.glob('*.mp4'))), 101)
            self.assertTrue(all((clips / row.file_path).read_bytes() == valid for row in rows))
            self.assertEqual(self.scope['time'].sleep.call_count, 1)

    def test_persistent_503_is_bounded_and_names_clip_without_exposing_url(self):
        url = 'https://example.invalid/video?signature=private-value'
        self.scope['urlopen'].side_effect = HTTPError(url, 503, 'Service Unavailable', {}, None)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'affected_clip.mp4'
            with self.assertRaisesRegex(RuntimeError, 'affected_clip.mp4: HTTP 503 after 5 attempts') as caught:
                self.scope['download_checked'](url, path, 'unused')
            self.assertNotIn('private-value', str(caught.exception))
            self.assertEqual(self.scope['urlopen'].call_count, 5)
            self.assertEqual([c.args[0] for c in self.scope['time'].sleep.call_args_list], [2, 4, 8, 16])
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_retry_after_is_respected_and_long_wait_stops_without_early_retry(self):
        url = 'https://example.invalid/video'
        valid = b'complete clip'
        digest = hashlib.sha256(valid).hexdigest()
        self.scope['urlopen'].side_effect = [
            HTTPError(url, 429, 'Too Many Requests', {'Retry-After': '7'}, None), io.BytesIO(valid)]
        with TemporaryDirectory() as tmp:
            self.scope['download_checked'](url, Path(tmp) / 'clip.mp4', digest)
        self.scope['time'].sleep.assert_called_once_with(7)
        self.scope['urlopen'].reset_mock()
        self.scope['time'].sleep.reset_mock()
        self.scope['urlopen'].side_effect = HTTPError(url, 503, 'Service Unavailable', {'Retry-After': '120'}, None)
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'wait of 120s'):
                self.scope['download_checked'](url, Path(tmp) / 'clip.mp4', digest)
        self.scope['urlopen'].assert_called_once()
        self.scope['time'].sleep.assert_not_called()

    def test_permanent_http_errors_are_not_retried(self):
        for status in [403, 404]:
            with self.subTest(status=status), TemporaryDirectory() as tmp:
                self.scope['urlopen'].reset_mock()
                self.scope['urlopen'].side_effect = HTTPError('https://example.invalid', status, 'Permanent error', {}, None)
                with self.assertRaises(HTTPError):
                    self.scope['download_checked']('https://example.invalid', Path(tmp) / 'clip.mp4', 'unused')
                self.scope['urlopen'].assert_called_once()
                self.scope['time'].sleep.assert_not_called()

    def test_connection_and_incomplete_read_failures_retry_without_partial_files(self):
        valid = b'complete clip'
        digest = hashlib.sha256(valid).hexdigest()
        failures = [URLError('connection reset'), TimeoutError(), ConnectionResetError()]
        broken_response = Mock()
        broken_response.__enter__ = Mock(return_value=broken_response)
        broken_response.__exit__ = Mock(return_value=False)
        broken_response.read.side_effect = IncompleteRead(b'partial bytes', 10)
        for failure in failures + [broken_response]:
            with self.subTest(failure=type(failure).__name__), TemporaryDirectory() as tmp:
                self.scope['urlopen'].reset_mock()
                self.scope['urlopen'].side_effect = [failure, io.BytesIO(valid)]
                path = Path(tmp) / 'clip.mp4'
                self.assertEqual(self.scope['download_checked']('https://example.invalid', path, digest), valid)
                self.assertEqual(path.read_bytes(), valid)
                self.assertEqual(list(Path(tmp).iterdir()), [path])
                self.assertEqual(self.scope['urlopen'].call_count, 2)

    def test_checksum_failure_is_not_retried_or_saved(self):
        self.scope['urlopen'].return_value = io.BytesIO(b'wrong bytes')
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'clip.mp4'
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                self.scope['download_checked']('https://example.invalid', path, hashlib.sha256(b'valid').hexdigest())
            self.assertFalse(path.exists())
        self.scope['urlopen'].assert_called_once()
        self.scope['time'].sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
