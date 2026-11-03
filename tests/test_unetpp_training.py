"""Behavioral CPU checks for full-data training, cache integrity and interrupted resume."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rd_ultrasound_ml.unetpp_training import (
    CheckpointStore, binary_metrics, capture_random_state, epoch_order, file_digest,
    restore_random_state, run_training, seed_training, train_epoch_accumulated,
    probe_training_precision,
)
from rd_ultrasound_ml.unetpp_data import (
    CachedClips, prepare_cached_clip, record_video_sources, verify_partitions,
)


def configuration():
    return {'experiment_id': 'synthetic_only', 'seed': 42,
        'training': {'epochs': 3, 'physical_batch': 2, 'effective_batch': 4,
        'precision': 'float32', 'learning_rate': .01, 'betas': [.9, .999],
        'eps': 1e-8, 'weight_decay': 0., 'checkpoint_every_updates': 1},
        'evaluation': {'threshold': .5}}


def tiny_model():
    # Dropout makes global RNG restoration observable in the resume test.
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Dropout(.2), nn.Linear(4, 1))


def fixture_dataset():
    return TensorDataset(torch.arange(21, dtype=torch.float32).reshape(7, 3) / 20,
                         torch.tensor([0., 1., 0., 1., 0., 0., 1.]))


class UNetPPTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_accumulation_matches_full_effective_batches_including_short_tail(self):
        seed_training(42)
        inputs = torch.randn(11, 3, dtype=torch.float64)
        labels = torch.tensor([0., 1., 0., 1., 0., 0., 1., 0., 1., 1., 0.], dtype=torch.float64)
        dataset = TensorDataset(inputs, labels)
        model = nn.Linear(3, 1).double()
        reference = deepcopy(model)
        order = epoch_order(11, 42, 1)
        opt = torch.optim.SGD(model.parameters(), lr=.03)
        ref_opt = torch.optim.SGD(reference.parameters(), lr=.03)
        progress = train_epoch_accumulated(model, dataset, order, opt, 'cpu', 2, 4)
        for start in range(0, 11, 4):
            chosen = order[start:start + 4]
            ref_opt.zero_grad()
            nn.BCEWithLogitsLoss()(reference(inputs[chosen]).reshape(-1), labels[chosen]).backward()
            ref_opt.step()
        self.assertEqual(progress['position'], 11)
        self.assertEqual(progress['optimizer_updates'], 3)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'boundary'):
            train_epoch_accumulated(model, dataset, order, opt, 'cpu', 2, 4, start_position=2)

    def test_mid_epoch_and_session_resume_match_uninterrupted_training(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings, dataset = configuration(), fixture_dataset()
            direct = run_training(tiny_model, dataset, dataset, settings, root/'direct', 'fixed', {},
                                  device='cpu', max_epochs_this_call=None, report=lambda _: None)
            original_save = CheckpointStore.save
            interrupted = False
            def save_then_disconnect(store, payload):
                nonlocal interrupted
                original_save(store, payload)
                if payload['phase'] == 'train' and payload['progress']['position'] == 4 and not interrupted:
                    interrupted = True
                    raise ConnectionError('synthetic Colab disconnect after committed update')
            with patch.object(CheckpointStore, 'save', save_then_disconnect), self.assertRaises(ConnectionError):
                run_training(tiny_model, dataset, dataset, settings, root/'resumed', 'fixed', {}, device='cpu')
            partial = run_training(tiny_model, dataset, dataset, settings, root/'resumed', 'fixed', {},
                                   device='cpu', max_epochs_this_call=1, report=lambda _: None)
            self.assertEqual(partial['status'], 'partial')
            self.assertEqual(partial['epochs_completed'], 1)
            resumed = run_training(tiny_model, dataset, dataset, settings, root/'resumed', 'fixed', {},
                                   device='cpu', max_epochs_this_call=None, report=lambda _: None)
            self.assertEqual(resumed['status'], 'completed')
            self.assertEqual(resumed['best_epoch'], direct['best_epoch'])
            first = CheckpointStore(root/'direct').load()
            second = CheckpointStore(root/'resumed').load()
            for name in first['model_state']:
                torch.testing.assert_close(first['model_state'][name], second['model_state'][name], rtol=0, atol=0)
            for key in ['scheduler_state', 'scaler_state', 'best_accuracy', 'best_epoch', 'order', 'progress']:
                self.assertEqual(first[key], second[key])
            for left, right in zip(first['history'], second['history']):
                for key in left.keys() - {'seconds_this_segment', 'validation_seconds', 'resumed_from_position'}:
                    self.assertEqual(left[key], right[key])
            best = torch.load(root/'resumed/best.pt', weights_only=True)
            self.assertEqual(best['epoch'], second['best_epoch'])
            for name in best['model_state']:
                torch.testing.assert_close(best['model_state'][name], second['best_model_state'][name])
            again = run_training(tiny_model, dataset, dataset, settings, root/'resumed', 'fixed', {},
                                 device='cpu', max_epochs_this_call=None, report=lambda _: None)
            self.assertEqual(again['epochs_completed'], 3)
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                run_training(tiny_model, dataset, dataset, settings, root/'resumed', 'changed', {}, device='cpu')
            changed = deepcopy(settings)
            changed['seed'] = 99
            with self.assertRaisesRegex(ValueError, 'configuration'):
                run_training(tiny_model, dataset, dataset, changed, root/'resumed', 'fixed', {}, device='cpu')

    def test_checkpoint_fallback_repairs_pointer_before_next_interrupted_write(self):
        with TemporaryDirectory() as tmp:
            store = CheckpointStore(tmp)
            store.save({'step': 1})
            store.save({'step': 2})
            pointer = json.loads(store.pointer.read_text())
            (Path(tmp)/pointer['file']).write_bytes(b'interrupted/corrupt')
            self.assertEqual(store.load(), {'step': 1})
            # Fail the next pointer commit after its new tensor file is written.
            with patch('rd_ultrasound_ml.unetpp_training.write_json_atomic', side_effect=OSError('disconnect')):
                with self.assertRaises(OSError):
                    store.save({'step': 3})
            self.assertEqual(store.load(), {'step': 1})
            store.save({'step': 4})
            self.assertEqual(store.load(), {'step': 4})

    def test_random_state_checkpoint_is_safe_load_compatible(self):
        seed_training(42)
        state = capture_random_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        with TemporaryDirectory() as tmp:
            store = CheckpointStore(tmp)
            store.save(state)
            restore_random_state(store.load())
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)

    def test_metrics_and_cpu_precision_guard(self):
        metrics = binary_metrics(np.array([0]*391+[1]*40), np.zeros(431))
        self.assertEqual((metrics['TP'], metrics['TN'], metrics['FP'], metrics['FN']), (0, 391, 0, 40))
        self.assertIsNone(metrics['precision'])
        self.assertAlmostEqual(metrics['accuracy'], 391/431)
        self.assertEqual(metrics['ROC_AUC'], .5)
        with patch.object(torch.cuda, 'is_available', return_value=False), self.assertRaisesRegex(RuntimeError, 'CUDA'):
            probe_training_precision(tiny_model, torch.zeros(2, 3), torch.zeros(2))

    def test_tensor_cache_reuses_verified_outputs_and_rejects_corruption(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root/'clips/fixture.avi'
            video.parent.mkdir()
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (48, 32))
            self.assertTrue(writer.isOpened())
            for value in [30, 100, 200]:
                writer.write(np.full((32, 48, 3), value, dtype=np.uint8))
            writer.release()
            row = SimpleNamespace(clip_id='fixture', file_path='fixture.avi', frame_count=3, height=32, width=48)
            source = {'sha256': file_digest(video), 'url': 'fixture', 'bytes': video.stat().st_size}
            calls = []
            def checked_download(url, path, digest, expected_bytes):
                self.assertEqual(file_digest(path), digest)
                calls.append(url)
            path, check = prepare_cached_clip(row, source, root, root/'cache', 'fixed', checked_download, target=(5, 8, 8))
            reused, second = prepare_cached_clip(row, source, root, root/'cache', 'fixed', checked_download, target=(5, 8, 8))
            self.assertEqual(path, reused)
            self.assertEqual(check, second)
            self.assertEqual(calls, ['fixture'])
            tensor, label = CachedClips([path], [1])[0]
            self.assertEqual(tuple(tensor.shape), (1, 5, 8, 8))
            self.assertEqual(label.item(), 1.)
            path.write_bytes(b'broken tensor')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                prepare_cached_clip(row, source, root, root/'cache', 'fixed', checked_download, target=(5, 8, 8))

    def test_source_catalog_chunk_persistence_and_no_missing_paths(self):
        with TemporaryDirectory() as tmp:
            frame = pd.DataFrame({'file_path': [f'non_rd/{index}.mp4' for index in range(51)]})
            class API:
                def __init__(self): self.calls = 0
                def get_paths_info(self, dataset, paths, **kwargs):
                    self.calls += 1
                    if self.calls == 2: raise ConnectionError('interrupted API connection')
                    return [SimpleNamespace(path=value, size=123, lfs=SimpleNamespace(size=123, sha256='a'*64)) for value in paths]
            api = API()
            path = Path(tmp)/'catalog.json'
            with self.assertRaises(ConnectionError):
                record_video_sources(frame, 'revision', path, api, report=lambda _: None)
            self.assertEqual(len(json.loads(path.read_text())['files']), 50)
            sources = record_video_sources(frame, 'revision', path, api, report=lambda _: None)
            self.assertEqual(len(sources), 51)
            self.assertEqual(api.calls, 3)

    def test_partition_validation_preserves_order_and_rejects_wrong_labels(self):
        train = pd.DataFrame({'path': [f'train/{i}' for i in range(3873)], 'label': [0]*3511+[1]*362})
        val = pd.DataFrame({'path': [f'val/{i}' for i in range(431)], 'label': [0]*391+[1]*40})
        all_rows = pd.concat([train, val], ignore_index=True)
        metadata = pd.DataFrame({'file_path': all_rows['path'], 'clip_id': range(4304),
            'diagnostic_class': all_rows['label'].map({0: 'non_rd', 1: 'rd'})})
        joined, _ = verify_partitions(train, val, metadata)
        self.assertEqual(joined['path'].tolist(), train['path'].tolist())
        metadata.loc[0, 'diagnostic_class'] = 'rd'
        with self.assertRaisesRegex(ValueError, 'disagree'):
            verify_partitions(train, val, metadata)

    def test_notebook_modules_config_and_reserved_test_agree(self):
        notebook = json.loads((ROOT/'notebooks/unetpp_training.ipynb').read_text())
        cells = {cell['id']: ''.join(cell['source']) for cell in notebook['cells']}
        config = ast.literal_eval(ast.parse(cells['training-config']).body[0].value)
        self.assertEqual(config, json.loads((ROOT/'configs/unetpp_rd_001_training.json').read_text()))
        for name in ['unetpp_baseline', 'unetpp_training', 'unetpp_data']:
            source = ast.literal_eval(ast.parse(cells['training-embed-'+name]).body[0].value)
            self.assertEqual(source, (ROOT/f'src/rd_ultrasound_ml/{name}.py').read_text())
        opened = [node.args[0].value for node in ast.walk(ast.parse(cells['training-partitions']))
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name) and node.func.value.id == 'unetpp_archive'
                  and node.func.attr == 'read']
        self.assertEqual(opened, ['splits/non_rd_vs_rd/train.csv', 'splits/non_rd_vs_rd/val.csv'])
        self.assertEqual(config['training']['epochs'], 50)
        self.assertEqual(config['training']['physical_batch']*config['training']['gradient_accumulation'], 48)
        self.assertFalse(config['training']['class_weighting'])
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), '<notebook>', 'exec')


if __name__ == '__main__':
    unittest.main()
