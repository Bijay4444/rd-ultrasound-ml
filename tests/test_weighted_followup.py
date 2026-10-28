"""Controlled weighted-loss comparison on synthetic data, without Colab or ERDES."""
import ast
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import random
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


class WeightedFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads((ROOT / 'notebooks/baseline_training.ipynb').read_text())
        cls.cells = {c['id']: ''.join(c['source']) for c in notebook['cells']}
        cls.base_config = json.loads((ROOT / 'configs/baseline_001_training.json').read_text())
        cls.scope = {'np': np, 'pd': pd, 'torch': torch, 'nn': nn, 'json': json, 'os': os,
                     'Path': Path, 'hashlib': hashlib, 'time': time, 'zipfile': zipfile, 'random': random,
                     'Dataset': Dataset, 'DataLoader': DataLoader, 'datetime': datetime,
                     'timezone': timezone, 'average_precision_score': average_precision_score,
                     'roc_auc_score': roc_auc_score, 'device': torch.device('cpu')}
        cls.definition_code = []
        for ident in ['pilot-runtime', 'pilot-config', 'pilot-model-metrics', 'pilot-data', 'followup-functions']:
            definitions = [n for n in ast.parse(cls.cells[ident]).body
                           if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
            code = compile(ast.Module(body=definitions, type_ignores=[]), ident, 'exec')
            cls.definition_code.append(code)
            exec(code, cls.scope)
        torch.set_num_threads(2)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    def test_weight_uses_training_counts_and_rejects_missing_classes(self):
        weight = self.scope['training_positive_weight']
        self.assertAlmostEqual(weight(np.array([0] * 232 + [1] * 24)), 232 / 24)
        self.assertEqual(weight([0, 0, 1]), 2.)
        for labels in [[], [0, 0], [1], [0, 2]]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                weight(labels)

    def test_proposal_changes_training_loss_while_preserving_other_settings(self):
        original = json.loads(json.dumps(self.base_config))
        actual = self.scope['make_weighted_config'](self.base_config, [0] * 232 + [1] * 24)
        tracked = json.loads((ROOT / 'configs/baseline_002_weighted.json').read_text())
        self.assertEqual(actual, tracked)
        self.assertEqual(self.base_config, original)
        for key in ['seed', 'preprocessing', 'model', 'evaluation', 'split_policy']:
            self.assertEqual(actual[key], original[key])
        for key, value in original['training'].items():
            if key != 'loss':
                self.assertEqual(actual['training'][key], value)
        tree = ast.parse(self.cells['weighted-config'])
        self.assertEqual(ast.literal_eval(tree.body[0].value), tracked)

    def test_positive_weight_increases_positive_loss_without_changing_negative_loss(self):
        logits = torch.tensor([0., 0.], requires_grad=True)
        labels = torch.tensor([0., 1.])
        plain = nn.BCEWithLogitsLoss(reduction='none')(logits, labels)
        weighted = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([232 / 24]), reduction='none')(logits, labels)
        self.assertAlmostEqual(weighted[0].item(), plain[0].item())
        self.assertAlmostEqual(weighted[1].item() / plain[1].item(), 232 / 24, places=5)
        weighted.mean().backward()
        self.assertGreater(abs(logits.grad[1].item()), abs(logits.grad[0].item()))

    def test_audit_detects_misaligned_or_changed_predictions(self):
        manifest = pd.DataFrame({'clip_id': ['a', 'b'], 'file_path': ['a.mp4', 'b.mp4'], 'label': [0, 1]})
        predictions = manifest.rename(columns={'label': 'dataset_label'}).copy()
        predictions['rd_probability'] = [.1, .4]
        predictions['predicted_label'] = [0, 0]
        reported = self.scope['classification_metrics']([0, 1], [.1, .4])
        self.assertEqual(self.scope['audit_saved_predictions'](predictions, manifest, reported, .5), reported)
        with self.assertRaises(ValueError):
            self.scope['audit_saved_predictions'](predictions.iloc[::-1], manifest, reported, .5)
        altered = dict(reported, average_precision=.1)
        with self.assertRaisesRegex(ValueError, 'average_precision'):
            self.scope['audit_saved_predictions'](predictions, manifest, altered, .5)

    def test_validation_uses_explicit_unweighted_loss_and_preserves_parameters(self):
        model = nn.Sequential(nn.Linear(2, 1), nn.Flatten(0))
        inputs, labels = torch.zeros(4, 2), torch.tensor([0., 0., 0., 1.])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
        before = [p.detach().clone() for p in model.parameters()]
        with torch.no_grad():
            expected = nn.BCEWithLogitsLoss()(model(inputs), labels).item()
        self.scope['criterion'] = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.]))
        actual, order, _ = self.scope['run_comparison_epoch'](model, loader, nn.BCEWithLogitsLoss())
        self.assertAlmostEqual(actual, expected, places=6)
        np.testing.assert_array_equal(order, labels.numpy())
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_actual_followup_cells_train_reload_export_without_changing_baseline(self):
        with TemporaryDirectory(dir=ROOT / 'outputs') as tmp, redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            baseline_dir = directory / 'baseline'
            baseline_dir.mkdir()
            scope = self.scope.copy()
            config = json.loads(json.dumps(self.base_config))
            config['training'].update(epochs=2, batch_size=4)
            labels = np.array([0] * 6 + [1] * 2)
            pilot = pd.DataFrame({'clip_id': [f'train{i}' for i in range(8)],
                'file_path': [f'train{i}.mp4' for i in range(8)], 'label': labels})
            validation = pd.DataFrame({'clip_id': [f'val{i}' for i in range(8)],
                'file_path': [f'val{i}.mp4' for i in range(8)], 'label': labels})
            predictions = validation.rename(columns={'label': 'dataset_label'}).copy()
            predictions['rd_probability'] = np.linspace(.05, .45, 8)
            predictions['predicted_label'] = 0
            metrics = scope['classification_metrics'](labels, predictions['rd_probability'])
            reference = scope['classification_metrics'](labels, np.zeros(8))
            write = scope['write_followup_json']
            write(baseline_dir, 'config.json', config)
            predictions.to_csv(baseline_dir / 'best_validation_predictions.csv', index=False)
            for name, frame in [('train_manifest.csv', pilot), ('validation_manifest.csv', validation)]:
                frame.to_csv(baseline_dir / name, index=False, lineterminator='\n')
            hashes = {name: scope['sha256']((baseline_dir / name).read_bytes())
                      for name in ['train_manifest.csv', 'validation_manifest.csv']}
            summary = {'experiment_id': 'baseline_001_pilot', 'test_used': False,
                'config_sha256': scope['sha256']((baseline_dir / 'config.json').read_bytes()),
                'manifest_hashes': hashes, 'best_validation_metrics': {'epoch': 1, 'validation_loss': .7, **metrics},
                'constant_non_rd_reference': reference}
            write(baseline_dir, 'summary.json', summary)
            write(baseline_dir, 'run_status.json', {'status': 'completed'})
            original_bytes = {p.name: p.read_bytes() for p in baseline_dir.iterdir()}
            rng = np.random.default_rng(42)
            scope.update(output_dir=baseline_dir, config=config, pilot=pilot, validation=validation,
                train_array=rng.random((8, 1, 16, 128, 128), dtype=np.float32),
                val_array=rng.random((8, 1, 16, 128, 128), dtype=np.float32),
                display=lambda value: None, runtime={'verification': 'synthetic CPU'},
                FileLink=lambda path: path)
            for code in self.definition_code:
                exec(code, scope)
            exec(compile(self.cells['followup-audit'], 'followup-audit', 'exec'), scope)
            weighted_dir = directory / 'weighted'
            weighted_dir.mkdir()
            weighted_config = scope['make_weighted_config'](config, labels)
            write(weighted_dir, 'config.json', weighted_config)
            scope.update(weighted_dir=weighted_dir, weighted_config=weighted_config,
                weighted_config_hash=scope['sha256']((weighted_dir / 'config.json').read_bytes()))
            exec(compile(self.cells['weighted-fit'], 'weighted-fit', 'exec'), scope)
            checkpoint = torch.load(weighted_dir / 'best.pt', map_location='cpu', weights_only=True)
            restored = scope['make_model']()
            restored.load_state_dict(checkpoint['model_state_dict'])
            restored.eval()
            with torch.no_grad():
                probabilities = torch.sigmoid(restored(torch.from_numpy(scope['val_array']))).numpy()
            saved = pd.read_csv(weighted_dir / 'best_validation_predictions.csv')
            np.testing.assert_allclose(probabilities, saved['rd_probability'], rtol=1e-5, atol=1e-6)
            exec(compile(self.cells['weighted-export'], 'weighted-export', 'exec'), scope)
            with zipfile.ZipFile(scope['weighted_export_path']) as archive:
                self.assertTrue(any(name.endswith('/comparison.csv') for name in archive.namelist()))
                self.assertTrue(all(Path(name).suffix in {'.json', '.csv', '.pt'} for name in archive.namelist()))
            self.assertEqual(original_bytes, {p.name: p.read_bytes() for p in baseline_dir.iterdir()})
            self.assertFalse(json.loads((weighted_dir / 'summary.json').read_text())['test_used'])
            with self.assertRaisesRegex(RuntimeError, 'training artifacts'):
                exec(compile(self.cells['weighted-fit'], 'weighted-fit', 'exec'), scope)


if __name__ == '__main__':
    unittest.main()
