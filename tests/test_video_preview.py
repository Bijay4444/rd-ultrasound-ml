"""Playback regression checks using synthetic bytes, without dataset access."""
import ast
import base64
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from IPython.display import Video


class VideoPreviewTests(unittest.TestCase):
    def render_playback(self, constructor):
        notebook_path = Path(__file__).resolve().parents[1] / 'notebooks/video_inspection.ipynb'
        notebook = json.loads(notebook_path.read_text())
        cell = next(cell for cell in notebook['cells'] if cell['id'] == 'video-preview')
        # Exercise the notebook's actual playback loop, independently of frame plotting.
        loop = ast.parse(''.join(cell['source'])).body[-1]
        self.assertIsInstance(loop, ast.For)
        code = compile(ast.Module(body=[loop], type_ignores=[]), str(notebook_path), 'exec')

        with TemporaryDirectory(prefix='erdes-video-preview-') as tmp:
            root = Path(tmp)
            (root / 'clips').mkdir()
            rows = []
            payloads = []
            for index in range(2):
                payload = f'synthetic video bytes {index}'.encode()
                relative_path = f'clip_{index}.mp4'
                (root / 'clips' / relative_path).write_bytes(payload)
                payloads.append(payload)
                rows.append(SimpleNamespace(clip_id=f'clip_{index}', subtype='normal', file_path=relative_path))

            rendered = []
            context = {
                'data_dir': root,
                'sample': SimpleNamespace(itertuples=lambda index=False: iter(rows)),
                'Video': constructor,
                'display': lambda video: rendered.append(video._repr_html_()),
            }
            with redirect_stdout(io.StringIO()):
                exec(code, context)

        self.assertEqual(len(rendered), len(payloads))
        for html, payload in zip(rendered, payloads):
            self.assertIn('data:video/mp4;base64,', html)
            self.assertIn(base64.b64encode(payload).decode(), html)
            self.assertIn('controls', html)

    def test_playback_embeds_each_clip(self):
        self.render_playback(Video)

    def test_playback_supports_colab_constructor_path_check(self):
        def colab_video(data=None, **kwargs):
            # Replay the unconditional path check in the user's saved Colab traceback.
            os.path.exists(data)
            return Video(data=data, **kwargs)

        self.render_playback(colab_video)


if __name__ == '__main__':
    unittest.main()
