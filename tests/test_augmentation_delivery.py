"""Config inheritance, standalone notebook contracts and synthetic CPU delivery."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import build_augmentation_notebook
from prepare_augmentation_job import prepare, INPUTS, DOCKER_IMAGE
from rd_ultrasound_ml.video_augmentation import CANDIDATE_POLICY, TrainingAugmentedClips, run_augmented_training
from rd_ultrasound_ml.unetpp_training import file_digest, write_json_atomic
from rd_ultrasound_ml.comparison_export import prediction_rows, verify_best_checkpoint


class AugmentationDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_matched_config_and_unchanged_cached_preprocessing(self):
        original = json.loads((ROOT / 'configs/unetpp_rd_002_training.json').read_text())
        augmented = json.loads((ROOT / 'configs/unetpp_rd_aug_001_training.json').read_text())
        for key in ('seed', 'model', 'training', 'preprocessing', 'dataset_revision', 'task', 'label_mapping'):
            self.assertEqual(original[key], augmented[key], key)
        self.assertEqual(augmented['preprocessing']['augmentation'], 'none')
        self.assertEqual(augmented['training_augmentation'], CANDIDATE_POLICY)
        self.assertEqual(augmented['cache']['preparation_config_digest'],
                         json.loads((ROOT / 'configs/unetpp_rd_002_evaluation.json').read_text())['training_config_digest'])

    def test_standalone_notebook_is_current_and_has_exact_sources(self):
        import nbformat
        generated = build_augmentation_notebook.notebook()
        self.assertEqual(generated, json.loads((ROOT / 'notebooks/kaggle_unetpp_augmentation.ipynb').read_text()))
        nbformat.validate(nbformat.from_dict(generated))
        for cell in generated['cells']:
            if cell['cell_type'] != 'code':
                continue
            tree = ast.parse(''.join(cell['source']))
            if cell['id'].startswith('embed-'):
                name, source = [ast.literal_eval(arg) for arg in tree.body[0].value.args]
                self.assertEqual(source, (ROOT / f'src/rd_ultrasound_ml/{name}.py').read_text())
            self.assertEqual(cell['outputs'], [])
        frozen = json.loads((ROOT / 'configs/unetpp_rd_002_evaluation.json').read_text())
        for name, digest in frozen['training_implementation_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / f'src/rd_ultrasound_ml/{name}.py').read_bytes()).hexdigest(), digest)

    def test_private_bundle_is_pinned_idempotent_and_refuses_wrong_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            first = prepare('example', directory)
            self.assertEqual(first, prepare('example', directory))
            metadata = json.loads((first / 'kernel-metadata.json').read_text())
            self.assertTrue(metadata['is_private'])
            self.assertFalse(metadata['enable_internet'])
            self.assertEqual(metadata['kernel_sources'], INPUTS)
            self.assertEqual(metadata['docker_image'], DOCKER_IMAGE)
            self.assertIn('@sha256:', metadata['docker_image'])
            second = prepare('example', directory, 2, 2, 'example/unetpp-rd-aug-001-train-01/1')
            self.assertEqual(len(json.loads((second / 'kernel-metadata.json').read_text())['kernel_sources']), 3)
            for previous in ['example/unetpp-rd-002-train-03/1', 'example/unetpp-rd-aug-001-train-01', None]:
                with self.assertRaises(ValueError):
                    prepare('example', directory, 2, 1, previous)
            with self.assertRaisesRegex(ValueError, 'differs'):
                prepare('example', directory, epochs=2)

    def test_runtime_drift_stops_at_identity_check(self):
        generated = build_augmentation_notebook.notebook()
        cells = {c['id']: ''.join(c['source']) for c in generated['cells']}
        config = json.loads((ROOT / 'configs/unetpp_rd_aug_001_training.json').read_text())
        runtime = deepcopy(config['reference_runtime'])
        runtime['python'] = 'changed managed runtime'
        sources = {name: (ROOT / f'src/rd_ultrasound_ml/{name}.py').read_text()
            for name in build_augmentation_notebook.MODULES}
        namespace = {'AUG_CONFIG': config, 'unetpp_runtime': runtime,
            'unetpp_sources_embedded': sources, 'hashlib': hashlib}
        with self.assertRaisesRegex(RuntimeError, 'Runtime python differs'):
            exec(compile(cells['imports-and-identity'], '<runtime-guard>', 'exec'), namespace)

    def test_archive_verification_rejects_corruption_before_extracting(self):
        from verify_augmentation_results import verify
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'results.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('unetpp_rd_aug_001_amp/artifact_sha256.json',
                    json.dumps({'data.json': hashlib.sha256(b'original').hexdigest()}))
                bundle.writestr('unetpp_rd_aug_001_amp/data.json', b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Artifact hash differs'):
                verify(archive)
            self.assertFalse((Path(directory) / 'verified').exists())

    def test_actual_training_validation_export_cells_on_synthetic_cpu_data(self):
        # Adapt only device strings and the fixture model/config. Exercise the
        # real notebook training, selected-checkpoint reload, figures and ZIP.
        generated = build_augmentation_notebook.notebook()
        cells = {c['id']: ''.join(c['source']) for c in generated['cells']}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'synthetic'
            output.mkdir()
            x = torch.rand(7, 1, 4, 12, 12, generator=torch.Generator().manual_seed(17))
            labels = torch.tensor([0., 1., 0., 1., 0., 1., 0.])
            clean = TensorDataset(x, labels)
            before = x.clone()
            config = {'seed': 42, 'experiment_id': 'synthetic', 'training_augmentation': deepcopy(CANDIDATE_POLICY),
                'training': {'epochs': 2, 'physical_batch': 2, 'effective_batch': 4, 'precision': 'float32',
                    'learning_rate': .01, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0., 'checkpoint_every_updates': 1},
                'evaluation': {'threshold': .5}}
            def model():
                return nn.Sequential(nn.Conv3d(1, 2, 1), nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(2, 1))
            ns = {'AUG_CONFIG': config, 'AUG_EPOCHS_THIS_SESSION': 1, 'AUG_MAX_SESSION_SECONDS': 3600,
                'UNetPPRDClassifier': model, 'aug_train_dataset': TrainingAugmentedClips(clean,
                    [str(i) for i in range(7)], CANDIDATE_POLICY, 42), 'aug_validation_dataset': clean,
                'aug_validation': pd.DataFrame({'clip_id': [str(i) for i in range(7)], 'label': labels.numpy()}),
                'aug_output': output, 'aug_fingerprint': 'synthetic', 'unetpp_runtime': {},
                'run_augmented_training': run_augmented_training, 'verify_best_checkpoint': verify_best_checkpoint,
                'prediction_rows': prediction_rows, 'file_digest': file_digest, 'write_json_atomic': write_json_atomic,
                'json': json, 'pd': pd, 'time': time, 'zipfile': zipfile, 'display': lambda _: None}
            exec(compile(cells['training'].replace("device='cuda'", "device='cpu'"), '<training>', 'exec'), ns)
            exec(compile(cells['validation-exports'].replace("'cuda',", "'cpu',"), '<validation-export>', 'exec'), ns)
            exec(compile(cells['export'], '<export>', 'exec'), ns)
            self.assertEqual(json.loads((output / 'summary.json').read_text())['status'], 'partial')
            self.assertTrue(json.loads((output / 'best_checkpoint_reload.json').read_text())['passed'])
            self.assertEqual(len(pd.read_csv(output / 'best_validation_predictions.csv')), 7)
            for name, digest in json.loads((output / 'artifact_sha256.json').read_text()).items():
                self.assertEqual(file_digest(output / name), digest)
            with zipfile.ZipFile(Path(directory) / 'unetpp_rd_aug_001_amp_results.zip') as archive:
                self.assertIn('synthetic/learning_curves.png', archive.namelist())
                self.assertIsNone(archive.testzip())
            self.assertTrue(torch.equal(x, before))


if __name__ == '__main__':
    unittest.main()
