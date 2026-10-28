"""Pilot behavior on synthetic inputs; no ERDES data, network, or GPU required."""
import ast
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


class BaselineTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((ROOT / 'notebooks/baseline_training.ipynb').read_text())
        cls.cells = {cell['id']: ''.join(cell['source']) for cell in cls.notebook['cells']}
        cls.config = json.loads((ROOT / 'configs/baseline_001_training.json').read_text())
        cls.scope = {'np': np, 'pd': pd, 'nn': nn, 'torch': torch, 'Path': Path,
                     'PurePosixPath': __import__('pathlib').PurePosixPath,
                     'hashlib': hashlib, 'io': io, 'Dataset': Dataset,
                     'average_precision_score': average_precision_score,
                     'roc_auc_score': roc_auc_score, 'device': torch.device('cpu'),
                     'criterion': nn.BCEWithLogitsLoss(), 'config': cls.config}
        # Execute the notebook's definitions, without its downloads or GPU run.
        for ident in ['pilot-runtime', 'pilot-download-helper', 'pilot-selection-functions', 'pilot-model-metrics',
                      'pilot-data', 'pilot-training-functions']:
            tree = ast.parse(cls.cells[ident])
            definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                           or (ident == 'pilot-download-helper' and isinstance(node, (ast.Import, ast.ImportFrom)))]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), ident, 'exec'), cls.scope)
        torch.set_num_threads(2)

    def test_embedded_config_and_preprocessing_match_tracked_sources(self):
        tree = ast.parse(self.cells['pilot-config'])
        embedded = ast.literal_eval(tree.body[0].value)
        self.assertEqual(embedded, self.config)
        prep = json.loads((ROOT / 'notebooks/baseline_preparation.ipynb').read_text())
        functions = ''.join(next(c for c in prep['cells'] if c['id'] == 'prep-functions')['source'])
        self.assertEqual(self.cells['pilot-preprocessing'], functions)

    def test_selection_is_stable_under_row_reordering_and_uses_training_only(self):
        frame = pd.DataFrame({'clip_id': [f'clip_{i}' for i in range(20)],
                              'label': [0] * 12 + [1] * 8})
        select = self.scope['select_pilot']
        first = select(frame, {'0': 5, '1': 2}, 42)
        second = select(frame.sample(frac=1, random_state=3), {'0': 5, '1': 2}, 42)
        self.assertEqual(first['clip_id'].tolist(), second['clip_id'].tolist())
        self.assertEqual(first['label'].value_counts().to_dict(), {0: 5, 1: 2})
        self.assertTrue(set(first['clip_id']).issubset(set(frame['clip_id'])))
        with self.assertRaises(ValueError):
            select(frame, {'1': 9}, 42)

    def test_partition_rejects_inconsistent_labels_and_unsafe_paths(self):
        content = b'path,label\na.mp4,1\nb.mp4,0\n'
        metadata = pd.DataFrame({'file_path': ['a.mp4', 'b.mp4'], 'clip_id': ['a', 'b'],
                                 'diagnostic_class': ['rd', 'non_rd']})
        join = self.scope['join_partition']
        self.assertEqual(len(join(content, metadata, {0: 1, 1: 1})), 2)
        wrong = metadata.copy()
        wrong.loc[0, 'diagnostic_class'] = 'non_rd'
        with self.assertRaisesRegex(ValueError, 'Label/metadata'):
            join(content, wrong, {0: 1, 1: 1})
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            join(content.replace(b'a.mp4', b'../a.mp4'),
                 metadata.replace('a.mp4', '../a.mp4'), {0: 1, 1: 1})

    def test_only_train_and_validation_csv_are_opened(self):
        tree = ast.parse(self.cells['pilot-manifests'])
        opened = [node.args[0].value for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name) and node.func.value.id == 'archive'
                  and node.func.attr == 'read']
        self.assertEqual(opened, ['splits/non_rd_vs_rd/train.csv', 'splits/non_rd_vs_rd/val.csv'])

    def test_confusion_metrics_and_average_precision_definition(self):
        metrics = self.scope['classification_metrics']([0, 0, 1, 1], [.1, .8, .6, .4])
        for name in ['TP', 'TN', 'FP', 'FN']:
            self.assertEqual(metrics[name], 1)
        for name in ['sensitivity', 'specificity', 'precision', 'F1', 'balanced_accuracy', 'accuracy']:
            self.assertEqual(metrics[name], .5)
        self.assertAlmostEqual(metrics['average_precision'], (1 / 2 + 2 / 3) / 2)
        self.assertEqual(metrics['ROC_AUC'], .5)

    def test_constant_reference_and_undefined_metrics(self):
        metrics = self.scope['classification_metrics']([0, 0, 0, 1], [0, 0, 0, 0])
        self.assertEqual((metrics['TN'], metrics['FN']), (3, 1))
        self.assertIsNone(metrics['precision'])
        self.assertIn('precision', metrics['undefined_reasons'])
        self.assertEqual(metrics['sensitivity'], 0)
        self.assertEqual(metrics['balanced_accuracy'], .5)
        self.assertEqual(metrics['average_precision'], .25)
        self.assertEqual(metrics['ROC_AUC'], .5)
        json.dumps(metrics, allow_nan=False)
        single = self.scope['classification_metrics']([0, 0], [.1, .2])
        self.assertIsNone(single['ROC_AUC'])
        self.assertIsNone(single['sensitivity'])
        with self.assertRaises(ValueError):
            self.scope['classification_metrics']([0, 1], [.2, float('nan')])

    def test_checkpoint_ties_keep_earliest_epoch(self):
        improves = self.scope['improves_average_precision']
        self.assertFalse(improves(.3, .3))
        self.assertFalse(improves(.2, .3))
        self.assertFalse(improves(float('nan'), .3))
        self.assertTrue(improves(.4, .3))

    def test_actual_model_shape_gradients_and_parameter_count(self):
        torch.manual_seed(42)
        model = self.scope['Compact3DCNN']()
        inputs = torch.rand(2, 1, 16, 128, 128)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 17585)
        loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([0., 1.]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters()), 0)

    def test_validation_preserves_weights_and_order_while_training_changes_weights(self):
        model = nn.Sequential(nn.Linear(2, 1), nn.Flatten(0))
        labels = torch.tensor([0., 1., 1., 0.])
        loader = DataLoader(TensorDataset(torch.randn(4, 2), labels), batch_size=2)
        before = [p.detach().clone() for p in model.parameters()]
        loss, actual_labels, probabilities = self.scope['run_epoch'](model, loader)
        self.assertTrue(np.isfinite(loss))
        np.testing.assert_array_equal(actual_labels, labels.numpy())
        self.assertEqual(len(probabilities), len(labels))
        self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))
        self.scope['run_epoch'](model, loader, torch.optim.Adam(model.parameters(), lr=.01))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_cached_corruption_stops_without_network_or_overwrite(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'clip.mp4'
            path.write_bytes(b'corrupt')
            no_network = Mock(side_effect=AssertionError('network access forbidden'))
            self.scope['urlopen'] = no_network
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                self.scope['download_checked']('https://unused.invalid', path, hashlib.sha256(b'valid').hexdigest())
            self.assertEqual(path.read_bytes(), b'corrupt')
            no_network.assert_not_called()

    def test_step7_rerun_rebuilds_partial_arrays_in_manifest_order(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            scope = self.scope.copy()
            pilot = pd.DataFrame({'clip_id': ['train0', 'train1'], 'file_path': ['train0.mp4', 'train1.mp4'],
                                  'frame_count': [2, 2], 'label': [0, 1]})
            validation = pd.DataFrame({'clip_id': ['val0', 'val1'], 'file_path': ['val0.mp4', 'val1.mp4'],
                                       'frame_count': [2, 2], 'label': [0, 1]})
            one = np.ones((1, 2, 4, 4), dtype=np.float32)
            zero = np.zeros_like(one)
            prepare = Mock(side_effect=[(one, np.array([0, 1])), ValueError('injected preprocessing failure')])
            scope.update(data_dir=base, output_dir=base / 'results', run_id='same-run',
                         pilot=pilot, validation=validation, diagnostics_passed=True,
                         params={'num_frames': 2, 'height': 4, 'width': 4},
                         record_sources=Mock(), acquire_videos=Mock(), save_json=Mock(),
                         prepare_clip=prepare, DataLoader=DataLoader,
                         time=__import__('time'))
            code = compile(self.cells['pilot-data'], 'pilot-data', 'exec')
            with self.assertRaisesRegex(ValueError, 'injected preprocessing failure'):
                exec(code, scope)
            self.assertTrue((base / 'prepared_pilots/same-run/train.npy').exists())
            prepare.side_effect = [(one, np.array([0, 1])), (zero, np.array([0, 1])),
                                   (zero, np.array([0, 1])), (one, np.array([0, 1]))]
            exec(code, scope)
            np.testing.assert_array_equal(scope['train_array'], np.stack([one, zero]))
            np.testing.assert_array_equal(scope['val_array'], np.stack([zero, one]))
            self.assertEqual(scope['pilot']['clip_id'].tolist(), ['train0', 'train1'])


if __name__ == '__main__':
    unittest.main()
