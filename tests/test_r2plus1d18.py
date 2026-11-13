"""R(2+1)D-18 comparison checks: model contract, cache adapter, resume/reload, notebook.

Synthetic CPU data only; no Kaggle account, network, private cache or CUDA.
These checks are code evidence, not ERDES model performance.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from rd_ultrasound_ml.r2plus1d18_baseline import (GRADCAM_LAYER, R2Plus1D18RDClassifier, describe_model,
    gradcam_layer)
from rd_ultrasound_ml.video_cache import load_cache_for_run
from rd_ultrasound_ml.comparison_export import prediction_rows, verify_best_checkpoint
from rd_ultrasound_ml.unetpp_data import CachedClips
from rd_ultrasound_ml.unetpp_kaggle import load_prepared_shards, shard_indices
from rd_ultrasound_ml.unetpp_training import file_digest, json_digest, run_training, write_json_atomic
import build_r2plus1d18_notebook
from prepare_r2plus1d18_job import write_bundle

CONFIG = json.loads((ROOT/'configs/r2plus1d18_rd_001_training.json').read_text())
UNETPP = json.loads((ROOT/'configs/unetpp_rd_002_training.json').read_text())


class R2Plus1D18ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = R2Plus1D18RDClassifier()

    def test_only_first_stem_convolution_changes_and_factorized_blocks_remain(self):
        stem = self.model.backbone.stem
        first, temporal = stem[0], stem[3]
        self.assertEqual((first.in_channels, first.out_channels, first.kernel_size, first.stride, first.padding),
                         (1, 45, (1, 7, 7), (1, 2, 2), (0, 3, 3)))
        self.assertIsNone(first.bias)
        self.assertEqual((temporal.in_channels, temporal.out_channels, temporal.kernel_size, temporal.padding),
                         (45, 64, (3, 1, 1), (1, 0, 0)))
        self.assertEqual([type(m) for m in stem], [nn.Conv3d, nn.BatchNorm3d, nn.ReLU] * 2)
        from torchvision.models.video.resnet import Conv2Plus1D
        factorized = [m for m in self.model.modules() if isinstance(m, Conv2Plus1D)]
        self.assertEqual(len(factorized), 16)  # two per BasicBlock, eight blocks
        self.assertEqual((self.model.backbone.fc.in_features, self.model.backbone.fc.out_features), (512, 1))
        self.assertEqual(describe_model(self.model)['parameter_count'], CONFIG['model']['expected_parameter_count'])
        self.assertEqual(CONFIG['model']['expected_parameter_count'], 31_300_638 - 2 * 45 * 7 * 7)

    def test_replacement_initialization_follows_builder_kaiming_fan_out_rule(self):
        weight = self.model.backbone.stem[0].weight.detach()
        expected = (2 / (45 * 7 * 7)) ** .5
        self.assertAlmostEqual(weight.mean().item(), 0., delta=.004)
        self.assertAlmostEqual(weight.std().item(), expected, delta=.1 * expected)
        torch.manual_seed(42)
        first = R2Plus1D18RDClassifier().state_dict()
        torch.manual_seed(42)
        second = R2Plus1D18RDClassifier().state_dict()
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_no_pretrained_weights_are_requested(self):
        with patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('download attempted')):
            R2Plus1D18RDClassifier()
        self.assertFalse(CONFIG['model']['pretrained'])

    def test_raw_logit_per_clip_finite_loss_and_nonzero_finite_gradients(self):
        inputs, labels = torch.rand(2, 1, 16, 32, 32), torch.tensor([0., 1.])
        logits = self.model(inputs)
        self.assertEqual(tuple(logits.shape), (2,))
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [p.grad for p in self.model.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertTrue(all(torch.count_nonzero(g).item() > 0 for g in gradients))

    def test_full_protocol_input_shape_and_gradcam_feature_layer(self):
        self.assertIs(gradcam_layer(self.model), self.model.backbone.layer4)
        self.assertEqual(GRADCAM_LAYER, 'backbone.layer4')
        captured = {}
        gradcam_layer(self.model).register_forward_hook(lambda m, i, o: captured.__setitem__('shape', tuple(o.shape)))
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.rand(1, 1, 96, 128, 128))
        self.assertEqual(tuple(logits.shape), (1,))
        self.assertEqual(captured['shape'], (1, 512, 12, 8, 8))

    def test_rejects_rgb_or_unbatched_input(self):
        with self.assertRaisesRegex(ValueError, 'grayscale'):
            self.model(torch.rand(1, 3, 8, 32, 32))
        with self.assertRaisesRegex(ValueError, 'grayscale'):
            self.model(torch.rand(1, 8, 32, 32))


class CacheAdapterTests(unittest.TestCase):
    """Fixture mirrors the U-Net++ CPU outputs: manifest plus the saved preparation config.json."""

    def make_shards(self, directory):
        frame = pd.DataFrame({'clip_id': list('abcd'), 'file_path': [c + '.mp4' for c in 'abcd'],
            'path': [c + '.mp4' for c in 'abcd'], 'label': [0, 1, 0, 1],
            'frame_count': [3] * 4, 'height': [32] * 4, 'width': [48] * 4})
        train, validation = frame.iloc[:2].reset_index(drop=True), frame.iloc[2:].reset_index(drop=True)
        preparation = {**{key: UNETPP[key] for key in ['experiment_id', 'dataset_revision', 'task',
            'label_mapping', 'preprocessing', 'model', 'seed']}, 'implementation_sha256': {'unetpp_baseline': 'f' * 64}}
        identity = {'preprocessing': 'fixture_float32'}
        roots = []
        for index in range(2):
            root = Path(directory)/f'unetpp-rd-002-inputs-0{index}'
            root.mkdir()
            roots.append(root)
            entries = []
            for position in shard_indices(4, index):
                row = frame.iloc[position]
                array = root/(row['clip_id'] + '.npy')
                np.save(array, np.full((1, 96, 128, 128), position / 4, dtype=np.float32))
                source = {'sha256': 'a' * 64}
                check = {'identity': {'signature': json_digest(identity), 'clip_id': row['clip_id'],
                    'file_path': row['file_path'], 'source_sha256': source['sha256'], 'frame_count': 3,
                    'height': 32, 'width': 48, 'target': [96, 128, 128]}, 'tensor_sha256': file_digest(array)}
                entries.append({'global_index': position, 'clip_id': row['clip_id'], 'label': int(row['label']),
                    'source': source, 'check': check, 'array': array.name})
            write_json_atomic(root/'config.json', preparation)
            write_json_atomic(root/'prepared_manifest.json', {'complete': True, 'shard_count': 2,
                'shard_index': index, 'config_digest': json_digest(preparation),
                'preprocessing_identity': identity, 'entries': entries})
        training = deepcopy(CONFIG)
        training['cache'] = {**training['cache'], 'preparation_config_digest': json_digest(preparation),
                             'expected_cache_fingerprint': None}
        return preparation, training, train, validation, roots

    def patched(self, train, validation):
        return patch('rd_ultrasound_ml.unetpp_kaggle.read_official_inputs', return_value=(train, validation))

    def test_new_training_identity_loads_verified_cache_and_binds_both_identities(self):
        with TemporaryDirectory() as tmp:
            preparation, training, train, validation, _ = self.make_shards(tmp)
            with self.patched(train, validation):
                with self.assertRaisesRegex(ValueError, 'configuration/code differs'):
                    load_prepared_shards(tmp, training, report=lambda _: None)  # the problem being solved
                reference = load_prepared_shards(tmp, preparation, report=lambda _: None)
                result = load_cache_for_run(tmp, training, report=lambda _: None)
                self.assertEqual([p.stem for p in result[2]], list('abcd'))
                record = result[4]
                self.assertEqual(record['cache_fingerprint'], reference[3])
                self.assertEqual(record['preparation_config_digest'], json_digest(preparation))
                self.assertEqual(record['training_config_digest'], json_digest(training))
                self.assertEqual(result[3], record['run_fingerprint'])
                self.assertNotEqual(result[3], reference[3])
                pinned = deepcopy(training)
                pinned['cache']['expected_cache_fingerprint'] = reference[3]
                self.assertEqual(load_cache_for_run(tmp, pinned, report=lambda _: None)[4]['cache_fingerprint'], reference[3])
                changed = deepcopy(training)
                changed['training']['physical_batch'] = 2
                self.assertNotEqual(load_cache_for_run(tmp, changed, report=lambda _: None)[3], result[3])

    def test_modified_preparation_config_or_pins_are_rejected(self):
        with TemporaryDirectory() as tmp:
            preparation, training, train, validation, roots = self.make_shards(tmp)
            with self.patched(train, validation):
                wrong_pin = deepcopy(training)
                wrong_pin['cache']['preparation_config_digest'] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'pinned preparation identity'):
                    load_cache_for_run(tmp, wrong_pin, report=lambda _: None)
                wrong_fingerprint = deepcopy(training)
                wrong_fingerprint['cache']['expected_cache_fingerprint'] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'cache fingerprint'):
                    load_cache_for_run(tmp, wrong_fingerprint, report=lambda _: None)
                unpinned = deepcopy(training)
                del unpinned['cache']
                with self.assertRaisesRegex(ValueError, 'must pin'):
                    load_cache_for_run(tmp, unpinned, report=lambda _: None)
                other_preprocessing = deepcopy(training)
                other_preprocessing['preprocessing']['num_frames'] = 16
                with self.assertRaisesRegex(ValueError, 'preprocessing differs'):
                    load_cache_for_run(tmp, other_preprocessing, report=lambda _: None)
                edited = dict(preparation, seed=7)
                write_json_atomic(roots[1]/'config.json', edited)
                with self.assertRaisesRegex(ValueError, 'manifest digest'):
                    load_cache_for_run(tmp, training, report=lambda _: None)

    def test_inherited_tensor_label_and_order_checks_still_run(self):
        with TemporaryDirectory() as tmp:
            _, training, train, validation, roots = self.make_shards(tmp)
            with self.patched(train, validation):
                (roots[1]/'c.npy').write_bytes(b'broken')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    load_cache_for_run(tmp, training, report=lambda _: None)
        with TemporaryDirectory() as tmp:
            _, training, train, validation, roots = self.make_shards(tmp)
            flipped = train.copy()
            flipped.loc[1, 'label'] = 0
            with self.patched(flipped, validation):
                with self.assertRaisesRegex(ValueError, 'identity/label'):
                    load_cache_for_run(tmp, training, report=lambda _: None)


class ResumeAndReloadTests(unittest.TestCase):
    """Real R(2+1)D-18 classifier on small synthetic clips so CPU checks stay fast."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        generator = np.random.default_rng(3)
        paths = []
        for index in range(12):
            path = root/f'clip_{index}.npy'
            np.save(path, generator.random((1, 8, 32, 32), dtype=np.float32))
            paths.append(path)
        labels = [0, 1] * 6
        self.train = CachedClips(paths[:8], labels[:8])
        self.validation = CachedClips(paths[8:], labels[8:])
        self.validation_manifest = pd.DataFrame({'clip_id': [f'v{i}' for i in range(4)], 'label': labels[8:]})
        self.config = deepcopy(CONFIG)
        self.config['training'].update(epochs=2, physical_batch=2, effective_batch=4, precision='float32',
                                       checkpoint_every_updates=1)
        self.runtime = {'device': 'cpu', 'test': True}

    def tearDown(self):
        self.tmp.cleanup()

    def run_dir(self, name, epochs_per_call):
        directory = Path(self.tmp.name)/name
        summary = None
        for _ in range(2 // epochs_per_call):
            summary = run_training(R2Plus1D18RDClassifier, self.train, self.validation, self.config, directory,
                'fingerprint', self.runtime, device='cpu', max_epochs_this_call=epochs_per_call, report=lambda _: None)
        return directory, summary

    def test_session_resume_matches_uninterrupted_training_and_checkpoint_reload_reproduces_validation(self):
        whole, whole_summary = self.run_dir('whole', 2)
        split, split_summary = self.run_dir('split', 1)
        self.assertEqual(whole_summary['status'], 'completed')
        self.assertEqual(split_summary['epochs_completed'], 2)
        history_whole = json.loads((whole/'history.json').read_text())
        history_split = json.loads((split/'history.json').read_text())
        for a, b in zip(history_whole, history_split):
            for key in ['train_loss', 'validation_loss', 'accuracy', 'learning_rate_used']:
                self.assertEqual(a[key], b[key], key)
        self.assertEqual(json.loads((whole/'validation_epoch_002.json').read_text()),
                         json.loads((split/'validation_epoch_002.json').read_text()))
        self.assertEqual(history_split[1]['resumed_from_position'], 0)
        result = verify_best_checkpoint(R2Plus1D18RDClassifier, split, self.validation, 'cpu', 2, 'float32',
                                        tolerance=1e-6)
        self.assertTrue(result['passed'])
        self.assertEqual(result['selected_epoch'], split_summary['best_epoch'])
        self.assertEqual(result['checkpoint_sha256'], file_digest(split/'best.pt'))
        saved = json.loads((split/'best_validation_predictions.json').read_text())
        rows = prediction_rows(self.validation_manifest, saved['dataset_labels'], saved['rd_probabilities'], .5, 'validation')
        self.assertEqual([r['clip_id'] for r in rows], ['v0', 'v1', 'v2', 'v3'])
        self.assertTrue(all(r['predicted_class'] == int(r['rd_probability'] >= .5) for r in rows))
        saved['rd_probabilities'] = [min(1., p + .01) for p in saved['rd_probabilities']]
        write_json_atomic(split/'best_validation_predictions.json', saved)
        with self.assertRaisesRegex(ValueError, 'did not reproduce'):
            verify_best_checkpoint(R2Plus1D18RDClassifier, split, self.validation, 'cpu', 2, 'float32', tolerance=1e-6)
        with self.assertRaisesRegex(ValueError, 'labels/order'):
            prediction_rows(self.validation_manifest.iloc[::-1], saved['dataset_labels'], saved['rd_probabilities'], .5, 'validation')

    def test_resume_refuses_changed_identity(self):
        directory = Path(self.tmp.name)/'first'
        run_training(R2Plus1D18RDClassifier, self.train, self.validation, self.config, directory,
            'fingerprint', self.runtime, device='cpu', max_epochs_this_call=1, report=lambda _: None)
        with self.assertRaisesRegex(ValueError, 'fingerprint differs'):
            run_training(R2Plus1D18RDClassifier, self.train, self.validation, self.config, directory,
                'other-cache', self.runtime, device='cpu', max_epochs_this_call=1, report=lambda _: None)
        changed = deepcopy(self.config)
        changed['training']['learning_rate'] = 1e-4
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            run_training(R2Plus1D18RDClassifier, self.train, self.validation, changed, directory,
                'fingerprint', self.runtime, device='cpu', max_epochs_this_call=1, report=lambda _: None)


class ConfigNotebookBundleTests(unittest.TestCase):
    def test_config_inherits_protocol_and_pins_verified_unetpp_cache_identity(self):
        for key in ['seed', 'dataset_revision', 'task', 'label_mapping', 'preprocessing']:
            self.assertEqual(CONFIG[key], UNETPP[key], key)
        for key, value in UNETPP['training'].items():
            self.assertEqual(CONFIG['training'][key], value, key)
        self.assertEqual(CONFIG['training']['epochs'], 50)
        self.assertEqual(CONFIG['evaluation']['threshold'], .5)
        self.assertEqual(CONFIG['experiment_id'], 'r2plus1d18_rd_001_amp')
        freeze = json.loads((ROOT/'configs/unetpp_rd_002_evaluation.json').read_text())
        self.assertEqual(CONFIG['cache']['preparation_config_digest'], freeze['training_config_digest'])
        self.assertEqual(CONFIG['cache']['expected_cache_fingerprint'], freeze['training_data_fingerprint'])
        self.assertEqual(CONFIG['cache']['preparation_experiment_id'], UNETPP['experiment_id'])

    def test_preparation_digest_recomputes_from_tracked_preparation_notebook(self):
        import hashlib
        notebook = json.loads((ROOT/'notebooks/kaggle_unetpp_preparation.ipynb').read_text())
        cells = {c['id']: ''.join(c['source']) for c in notebook['cells']}
        config = ast.literal_eval(ast.parse(cells['kaggle-config']).body[0].value)
        config['implementation_sha256'] = {name: hashlib.sha256(ast.literal_eval(
            ast.parse(cells['kaggle-embed-' + name]).body[0].value).encode()).hexdigest()
            for name in ['unetpp_baseline', 'unetpp_training', 'unetpp_data', 'unetpp_kaggle']}
        self.assertEqual(json_digest(config), CONFIG['cache']['preparation_config_digest'])

    def test_tracked_notebook_is_current_embeds_sources_and_never_opens_test_data(self):
        tracked = json.loads((ROOT/'notebooks/kaggle_r2plus1d18_training.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(tracked, json.loads(json.dumps(build_r2plus1d18_notebook.notebook())))
        cells = {c['id']: ''.join(c['source']) for c in tracked['cells']}
        self.assertEqual(ast.literal_eval(ast.parse(cells['kaggle-config']).body[0].value), CONFIG)
        for module in build_r2plus1d18_notebook.MODULES:
            embedded = ast.literal_eval(ast.parse(cells['kaggle-embed-' + module]).body[0].value)
            self.assertEqual(embedded, (ROOT/f'src/rd_ultrasound_ml/{module}.py').read_text(encoding='utf-8'))
        code = '\n'.join(s for i, s in cells.items() if not i.startswith('kaggle-embed-'))
        for cell in tracked['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')
        for forbidden in ['test.csv', 'download_checked', 'prepare_cached_clip', '/kaggle/working/unetpp']:
            self.assertFalse(forbidden in code, forbidden)
        self.assertIn("'torchvision': str(torchvision.__version__)", cells['r2p1-runtime'])
        self.assertFalse(tracked['metadata']['kaggle']['isInternetEnabled'])

    def test_private_version_pinned_gpu_bundles(self):
        inputs = ['owner/unetpp-rd-002-inputs-00/1', 'owner/unetpp-rd-002-inputs-01/1']
        with TemporaryDirectory() as tmp:
            first, meta = write_bundle('ExampleUser', 1, inputs, tmp)
            self.assertEqual(meta['kernel_sources'], inputs)
            self.assertTrue(meta['is_private'] and meta['enable_gpu'])
            self.assertFalse(meta['enable_internet'])
            self.assertEqual(meta['title'].lower().replace(' ', '-'), meta['id'].split('/')[1])
            _, second = write_bundle('ExampleUser', 2, inputs, tmp, epochs=10)
            self.assertEqual(second['kernel_sources'][-1], 'exampleuser/r2plus1d18-rd-001-train-01')
            notebook = json.loads((Path(tmp)/'r2plus1d18-rd-001-train-02/kaggle_r2plus1d18_training.ipynb').read_text())
            values = {}
            exec(''.join(next(c for c in notebook['cells'] if c['id'] == 'kaggle-settings')['source']), values)
            self.assertEqual((values['R2P1_EPOCHS_THIS_SESSION'], values['R2P1_REQUIRE_PREVIOUS']), (10, True))
            with self.assertRaisesRegex(ValueError, 'version-pinned'):
                write_bundle('ExampleUser', 1, ['owner/a', 'owner/b'], tmp)
            with self.assertRaisesRegex(ValueError, 'differs'):
                write_bundle('ExampleUser', 2, inputs, tmp, epochs=9)


if __name__ == '__main__':
    unittest.main()
