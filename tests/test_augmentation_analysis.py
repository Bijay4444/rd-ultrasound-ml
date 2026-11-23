"""Paired comparison alignment and prepared evaluation gates; no real test access."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from analyze_augmentation_results import paired_predictions
import prepare_augmentation_evaluation as evaluation


class AugmentationAnalysisTests(unittest.TestCase):
    def test_paired_error_categories_match_individual_decisions(self):
        manifest = pd.DataFrame({'clip_id': list('abcd'), 'label': [0, 1, 1, 0]})
        baseline = {'dataset_labels': [0, 1, 1, 0], 'rd_probabilities': [.1, .8, .2, .9]}
        augmented = {'dataset_labels': [0, 1, 1, 0], 'rd_probabilities': [.6, .3, .7, .1]}
        paired = paired_predictions(manifest, baseline, augmented)
        self.assertEqual(paired['outcome'].tolist(), ['baseline_correct_augmentation_wrong'] * 2 +
                         ['baseline_wrong_augmentation_correct'] * 2)
        self.assertEqual(paired['clip_id'].tolist(), list('abcd'))

    def test_misaligned_labels_or_invalid_scores_cannot_make_a_comparison(self):
        manifest = pd.DataFrame({'clip_id': ['a', 'b'], 'label': [0, 1]})
        good = {'dataset_labels': [0, 1], 'rd_probabilities': [.1, .9]}
        for bad in [{'dataset_labels': [1, 0], 'rd_probabilities': [.1, .9]},
                    {'dataset_labels': [0, 1], 'rd_probabilities': [float('nan'), .9]},
                    {'dataset_labels': [0, 1], 'rd_probabilities': [.1]}]:
            with self.assertRaises(ValueError):
                paired_predictions(manifest, good, bad)

    def test_prepared_notebook_is_current_and_requires_confirmation_before_test_access(self):
        freeze = json.loads(evaluation.FREEZE_PATH.read_text())
        generated = evaluation.notebook(freeze)
        self.assertEqual(generated, json.loads(evaluation.NOTEBOOK_PATH.read_text()))
        nbformat.validate(nbformat.from_dict(generated))
        for cell in generated['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')
                self.assertEqual(cell['outputs'], [])
        cells = {c['id']: ''.join(c['source']) for c in generated['cells']}
        pending = {**freeze, 'team_freeze_confirmed': False}
        with patch('rd_ultrasound_ml.comparison_evaluation.find_comparison_training') as find_training:
            with self.assertRaisesRegex(RuntimeError, 'no test data opened'):
                exec(cells['verify-before-test-access'], {'AUG_FREEZE': pending})
            find_training.assert_not_called()
        self.assertEqual(freeze['selected_epoch'], 32)
        self.assertEqual(freeze['threshold'], .5)
        self.assertFalse(freeze['test_accessed_at_freeze'])

    def test_submission_is_private_pinned_and_refuses_pending_confirmation(self):
        freeze = json.loads(evaluation.FREEZE_PATH.read_text())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Confirm the final team choices'):
                evaluation.write_bundle({**freeze, 'team_freeze_confirmed': False}, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
            synthetic_confirmation = {**deepcopy(freeze), 'team_freeze_confirmed': True,
                'team_freeze_note': 'Synthetic test fixture only; not an actual team decision'}
            output = evaluation.write_bundle(synthetic_confirmation, directory)
            metadata = json.loads((output / 'kernel-metadata.json').read_text())
            self.assertTrue(metadata['is_private'])
            self.assertFalse(metadata['enable_internet'])
            self.assertIn('@sha256:', metadata['docker_image'])
            self.assertEqual(metadata['kernel_sources'], [evaluation.SOURCE + '/1',
                'bijaykhadka1710/unetpp-rd-002-test-inputs/1'])


if __name__ == '__main__':
    unittest.main()
