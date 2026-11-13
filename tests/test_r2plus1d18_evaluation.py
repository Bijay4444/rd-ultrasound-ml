"""R(2+1)D-18 frozen reused-test evaluation: gates, cache reuse, inference and notebook (synthetic CPU)."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from rd_ultrasound_ml.comparison_evaluation import (
    verify_comparison_training, find_comparison_training, load_reused_test_cache, evaluate_comparison_model,
)
from rd_ultrasound_ml.unetpp_training import json_digest, file_digest, write_json_atomic
from build_r2plus1d18_evaluation_notebook import notebook as build_notebook, MODULES
from prepare_r2plus1d18_evaluation_job import write_evaluation_bundle

FREEZE_PATH = ROOT/'configs/r2plus1d18_rd_001_evaluation.json'
CLIPS = ['a', 'b', 'c', 'd']
LABELS = [0, 1, 0, 1]
MODULE = 'rd_ultrasound_ml.comparison_evaluation'


class R2Plus1D18EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def training_fixture(self, root):
        directory = Path(root)/'training'
        directory.mkdir()
        identity = {'preprocessing': {'fixture': True}, 'implementation': 'a'*64}
        config = {'experiment_id': 'synthetic', 'seed': 42, 'implementation_sha256': {'m': 'c'*64}}
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(5)
            model.bias.zero_()
        torch.save({'epoch': 15, 'config': config, 'data_fingerprint': 'run', 'model_state': model.state_dict(),
                    'test_used': False}, directory/'best.pt')
        write_json_atomic(directory/'config.json', config)
        write_json_atomic(directory/'summary.json', {'status': 'completed', 'epochs_completed': 50, 'best_epoch': 15,
            'config_digest': json_digest(config), 'data_fingerprint': 'run', 'test_used': False})
        write_json_atomic(directory/'input_fingerprint.json', {'preprocessing_identity': identity})
        preparation_freeze = {'evaluation_id': 'reference_prep'}
        freeze = {'team_freeze_confirmed': True, 'evaluation_id': 'synthetic_only', 'model_label': 'Synthetic',
            'training_experiment_id': 'synthetic', 'training_implementation_sha256': {'m': 'c'*64},
            'training_epochs_completed': 50, 'threshold': .5, 'test_accessed_at_freeze': False,
            'checkpoint_file': 'best.pt', 'checkpoint_sha256': file_digest(directory/'best.pt'),
            'training_config_digest': json_digest(config), 'training_run_fingerprint': 'run', 'selected_epoch': 15,
            'preprocessing_identity': identity, 'physical_batch': 2, 'precision': 'float32',
            'test_clips': 4, 'test_label_counts': {'0': 2, '1': 2},
            'test_cache': {'preparation_freeze_digest': json_digest(preparation_freeze)},
            'reference_test_rows': [{'source': 'reference model', 'partition': 'reused official test benchmark',
                                     'accuracy': .9}]}
        return directory, freeze, preparation_freeze

    def cache_fixture(self, root, freeze, preparation_freeze):
        cache = Path(root)/'input/test_inputs'
        (cache/'arrays').mkdir(parents=True)
        frame = pd.DataFrame({'clip_id': CLIPS, 'file_path': CLIPS, 'label': LABELS,
                              'frame_count': [3]*4, 'height': [32]*4, 'width': [48]*4})
        content = frame[['file_path', 'label']].to_csv(index=False).encode()
        with zipfile.ZipFile(cache/'splits.zip', 'w') as archive:
            archive.writestr('splits/non_rd_vs_rd/test.csv', content)
        (cache/'erdes_metadata.csv').write_text('file_path\n')
        identity = freeze['preprocessing_identity']
        entries = []
        for index, row in enumerate(frame.itertuples(index=False)):
            path = cache/'arrays'/(row.clip_id+'.npy')
            np.save(path, np.full((1, 96, 128, 128), index, np.float32))
            entries.append({'index': index, 'clip_id': row.clip_id, 'label': row.label, 'array': f'arrays/{path.name}',
                'source': {'sha256': 'b'*64}, 'check': {'tensor_sha256': file_digest(path), 'identity': {
                    'signature': json_digest(identity), 'clip_id': row.clip_id, 'file_path': row.file_path,
                    'source_sha256': 'b'*64, 'frame_count': 3, 'height': 32, 'width': 48, 'target': [96, 128, 128]}}})
        write_json_atomic(cache/'evaluation_freeze.json', preparation_freeze)
        write_json_atomic(cache/'test_prepared_manifest.json', {'complete': True,
            'freeze_digest': json_digest(preparation_freeze), 'test_csv_sha256': hashlib.sha256(content).hexdigest(),
            'preprocessing_identity': identity, 'entries': entries})
        return cache, frame

    def load(self, root, directory, freeze, frame):
        with patch(MODULE+'.read_official_inputs', return_value=(None, None)), \
                patch(MODULE+'.validate_test_partition', return_value=frame):
            return load_reused_test_cache(Path(root)/'input', directory, freeze, report=lambda _: None)

    def test_team_freeze_gate_blocks_before_any_test_access(self):
        with TemporaryDirectory() as tmp:
            directory, freeze, _ = self.training_fixture(tmp)
            with patch(MODULE+'.read_official_inputs') as read_split:
                with self.assertRaisesRegex(ValueError, 'Team has not confirmed'):
                    load_reused_test_cache('/must/not/read', directory, {**freeze, 'team_freeze_confirmed': False})
                read_split.assert_not_called()
            tracked = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
            if tracked['team_freeze_confirmed'] is not False:
                self.assertIs(tracked['team_freeze_confirmed'], True)
                self.assertIn('Confirmed', tracked['team_freeze_note'])

    def test_training_verification_rejects_changed_selection_checkpoint_or_code(self):
        with TemporaryDirectory() as tmp:
            directory, freeze, _ = self.training_fixture(tmp)
            verify_comparison_training(directory, freeze)
            self.assertEqual(find_comparison_training(tmp, freeze), directory)
            with self.assertRaisesRegex(ValueError, 'selected epoch'):
                verify_comparison_training(directory, {**freeze, 'selected_epoch': 50})
            with self.assertRaisesRegex(ValueError, 'implementation'):
                verify_comparison_training(directory, {**freeze, 'training_implementation_sha256': {'m': 'd'*64}})
            with self.assertRaisesRegex(ValueError, 'preprocessing'):
                verify_comparison_training(directory, {**freeze, 'preprocessing_identity': {'other': 1}})
            (directory/'best.pt').write_bytes(b'changed model')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                verify_comparison_training(directory, freeze)

    def test_reused_cache_accepts_valid_and_rejects_pin_identity_split_and_tensor_changes(self):
        with TemporaryDirectory() as tmp:
            directory, freeze, preparation_freeze = self.training_fixture(tmp)
            cache, frame = self.cache_fixture(tmp, freeze, preparation_freeze)
            loaded, dataset, record = self.load(tmp, directory, freeze, frame)
            self.assertEqual(len(dataset), 4)
            self.assertEqual(loaded['clip_id'].tolist(), CLIPS)
            self.assertEqual(record['preparation_freeze_digest'], freeze['test_cache']['preparation_freeze_digest'])
            with self.assertRaisesRegex(ValueError, 'preparation freeze'):
                self.load(tmp, directory, {**freeze, 'test_cache': {'preparation_freeze_digest': 'e'*64}}, frame)
            with self.assertRaisesRegex(ValueError, 'count/labels'):
                self.load(tmp, directory, {**freeze, 'test_label_counts': {'0': 3, '1': 1}}, frame)
            manifest = json.loads((cache/'test_prepared_manifest.json').read_text())
            write_json_atomic(cache/'test_prepared_manifest.json', {**manifest, 'test_csv_sha256': 'f'*64})
            with self.assertRaisesRegex(ValueError, 'official test split'):
                self.load(tmp, directory, freeze, frame)
            write_json_atomic(cache/'test_prepared_manifest.json',
                              {**manifest, 'preprocessing_identity': {'preprocessing': {'fixture': False}}})
            with self.assertRaisesRegex(ValueError, 'preprocessing identity'):
                self.load(tmp, directory, freeze, frame)
            write_json_atomic(cache/'test_prepared_manifest.json', manifest)
            (cache/'arrays/c.npy').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.load(tmp, directory, freeze, frame)

    def test_inference_uses_frozen_weights_threshold_and_records_timing(self):
        with TemporaryDirectory() as tmp:
            directory, freeze, _ = self.training_fixture(tmp)
            frame = pd.DataFrame({'clip_id': CLIPS, 'file_path': CLIPS, 'label': LABELS})
            dataset = TensorDataset(torch.tensor([[-1.], [1.], [-.5], [.5]]), torch.tensor([0., 1., 0., 1.]))
            output = Path(tmp)/'evaluation'
            before = file_digest(directory/'best.pt')
            with patch('torch.optim.Adam', side_effect=AssertionError('No optimizer may be constructed')):
                result = evaluate_comparison_model(lambda: torch.nn.Linear(1, 1), dataset, frame, directory, freeze,
                                                   output, {'gpu': None}, device='cpu', report=lambda _: None)
            metrics = result['metrics']
            self.assertEqual((metrics['TP'], metrics['TN'], metrics['FP'], metrics['FN']), (2, 2, 0, 0))
            self.assertEqual(file_digest(directory/'best.pt'), before)
            self.assertEqual(result['benchmark_label'], 'reused official test benchmark')
            self.assertFalse(result['training_performed'])
            self.assertFalse(result['threshold_tuned'])
            timing = result['inference_timing']
            self.assertEqual((timing['physical_batch'], timing['precision'], timing['clips']), (2, 'float32', 4))
            self.assertGreater(timing['model_forward_ms_per_clip_excluding_first_batch'], 0)
            prediction = pd.read_csv(output/'test_predictions.csv')
            self.assertEqual(prediction['clip_id'].tolist(), CLIPS)
            self.assertEqual(prediction['predicted_label'].tolist(), LABELS)
            comparison = pd.read_csv(output/'test_reference_comparison.csv')
            self.assertEqual(comparison['source'].tolist(), ['Synthetic (this evaluation)', 'reference model'])
            with self.assertRaisesRegex(ValueError, 'already exist'):
                evaluate_comparison_model(lambda: torch.nn.Linear(1, 1), dataset, frame, directory, freeze,
                                          output, {}, device='cpu')

    def test_tracked_freeze_pins_match_training_sources_and_reference_cache(self):
        freeze = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
        for name, digest in freeze['training_implementation_sha256'].items():
            source = (ROOT/f'src/rd_ultrasound_ml/{name}.py').read_text(encoding='utf-8')
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), digest, name)
        reference = json.loads((ROOT/'configs/unetpp_rd_002_evaluation.json').read_text())
        self.assertEqual(freeze['test_cache']['preparation_freeze_digest'], json_digest(reference))
        self.assertEqual(freeze['test_cache']['preparation_evaluation_id'], reference['evaluation_id'])
        self.assertEqual(freeze['preprocessing_identity']['preprocessing'], reference['preprocessing'])
        self.assertEqual((freeze['test_clips'], freeze['test_label_counts']),
                         (reference['test_clips'], reference['test_label_counts']))
        self.assertEqual((freeze['selected_epoch'], freeze['threshold']), (15, .5))

    def test_notebook_matches_builder_embeds_sources_and_has_no_training(self):
        notebook = json.loads((ROOT/'notebooks/kaggle_r2plus1d18_evaluation.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(notebook, build_notebook())
        cells = {cell['id']: ''.join(cell['source']) for cell in notebook['cells']}
        freeze = ast.literal_eval(ast.parse(cells['r2p1-eval-freeze']).body[0].value)
        self.assertEqual(freeze, json.loads(FREEZE_PATH.read_text(encoding='utf-8')))
        for name in MODULES:
            source = ast.literal_eval(ast.parse(cells['r2p1-eval-embed-'+name]).body[0].value)
            self.assertEqual(source, (ROOT/f'src/rd_ultrasound_ml/{name}.py').read_text(encoding='utf-8'))
        own = '\n'.join(text for ident, text in cells.items() if not ident.startswith('r2p1-eval-embed-'))
        for forbidden in ['run_training(', 'optim.', '.backward(', 'download']:
            self.assertNotIn(forbidden, own)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), 'evaluation', 'exec')

    def test_bundle_refuses_without_team_freeze_and_is_private_pinned_gpu(self):
        freeze = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
        owner = freeze['training_source'].split('/')[0]
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Team freeze not confirmed'):
                write_evaluation_bundle(owner, tmp, {**freeze, 'team_freeze_confirmed': False})
            self.assertFalse(any(Path(tmp).iterdir()))
            _, metadata = write_evaluation_bundle(owner, tmp, {**freeze, 'team_freeze_confirmed': True})
            self.assertTrue(metadata['is_private'])
            self.assertTrue(metadata['enable_gpu'])
            self.assertFalse(metadata['enable_internet'])
            self.assertEqual(metadata['kernel_sources'], ['namanbabariya/r2plus1d18-rd-001-train-04/1',
                                                          'bijaykhadka1710/unetpp-rd-002-test-inputs/1'])
            with self.assertRaisesRegex(ValueError, 'owner'):
                write_evaluation_bundle('someoneelse', tmp, {**freeze, 'team_freeze_confirmed': True})


if __name__ == '__main__':
    unittest.main()
