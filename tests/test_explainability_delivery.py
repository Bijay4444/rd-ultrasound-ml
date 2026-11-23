"""Notebook embedding/provenance and failure-export contract checks."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


class DeliveryTests(unittest.TestCase):
    def test_private_bundle_is_pinned_and_idempotent(self):
        sys.path.insert(0, str(ROOT/'scripts'))
        from prepare_explainability_job import prepare
        with tempfile.TemporaryDirectory() as directory:
            output = prepare('example', directory)
            self.assertEqual(prepare('example', directory), output)
            metadata = json.loads((output/'kernel-metadata.json').read_text())
            self.assertTrue(metadata['is_private'])
            self.assertFalse(metadata['enable_internet'])
            self.assertEqual(len(metadata['kernel_sources']), 4)
            self.assertTrue(all(source.rsplit('/', 1)[1].isdigit() for source in metadata['kernel_sources']))
            self.assertFalse(any('test' in source for source in metadata['kernel_sources']))

    def test_presentation_exports(self):
        import numpy as np
        import torch
        from PIL import Image
        from rd_ultrasound_ml.explainability_workflow import render_case
        x = torch.zeros(1, 1, 96, 128, 128)
        cam = np.zeros((96, 128, 128), dtype=np.float32)
        cam[:, 32:64, 32:64] = np.linspace(0, 1, 96)[:, None, None]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render_case(root, x, [{'normalized': cam}, {'normalized': 1-cam}], 'Synthetic export check')
            for name in ('panel.png', 'temporal.png', 'rd_overlay.gif', 'non_rd_overlay.gif'):
                with Image.open(root/name) as image:
                    self.assertGreater(image.width, 0)
                    if name.endswith('.gif'): self.assertEqual(image.n_frames, 12)

    def test_notebook_embeds_current_sources_and_freezes(self):
        spec = importlib.util.spec_from_file_location('explanation_builder', ROOT/'scripts/build_explainability_notebook.py')
        builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        generated = builder.notebook()
        self.assertEqual(json.loads((ROOT/'notebooks/kaggle_video_explainability.ipynb').read_text()), generated)
        embedded = {}
        for cell in generated['cells']:
            if cell['cell_type'] != 'code': continue
            tree = ast.parse(''.join(cell['source']))
            if cell['id'].startswith('embed-'):
                call = tree.body[0].value
                name, source = [ast.literal_eval(a) for a in call.args]
                self.assertEqual(source, (ROOT/f'src/rd_ultrasound_ml/{name}.py').read_text())
                embedded[name] = source
        for filename in ('unetpp_rd_002_evaluation.json', 'r2plus1d18_rd_001_evaluation.json'):
            freeze = json.loads((ROOT/'configs'/filename).read_text())
            for name, digest in freeze['training_implementation_sha256'].items():
                self.assertEqual(hashlib.sha256(embedded[name].encode()).hexdigest(), digest)

    def test_failed_job_exports_status_and_valid_hashes(self):
        from rd_ultrasound_ml.explainability_workflow import run_explanations
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'job'
            from rd_ultrasound_ml.video_explainability import ModelAdapter
            adapters = [ModelAdapter('unetpp', None, 'encoder.4', 'head', {})]
            with patch('torch.cuda.is_available', return_value=False):
                with self.assertRaises(RuntimeError): run_explanations(directory, output, adapters, {})
            self.assertEqual(json.loads((output/'status.json').read_text())['status'], 'failed')
            hashes = json.loads((output/'artifact_sha256.json').read_text())
            for name, digest in hashes.items():
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(), digest)
            with zipfile.ZipFile(output.with_suffix('.zip')) as archive:
                self.assertIn('job/status.json', archive.namelist())
            with self.assertRaises(FileExistsError): run_explanations(directory, output, [], {})


if __name__ == '__main__': unittest.main()
