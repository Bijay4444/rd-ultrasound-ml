"""No-network continuation checks: failures must never launch a later job."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import continue_augmentation_training as controller


def initial_state():
    return {'status': 'running', 'config_sha256': 'fixed', 'username': 'example',
        'epochs_completed': 1, 'last_session': 1, 'previous_source': 'example/unetpp-rd-aug-001-train-01/2',
        'last_epoch_seconds': 488, 'current_job': None, 'completed_jobs': [], 'test_used': False}


class ContinuationTests(unittest.TestCase):
    def setup_fakes(self, directory, statuses, verifications):
        events = []
        def prepare(username, output_root, session, epochs, previous):
            events.append(('prepare', session, epochs, previous))
            path = Path(output_root) / f'session_{session}'
            path.mkdir(parents=True, exist_ok=True)
            (path / 'kernel-metadata.json').write_text(json.dumps({'id': f'{username}/unetpp-rd-aug-001-train-{session:02d}'}))
            return path
        def execute(command):
            if 'list' in command:
                return 'ref,title\n'
            if 'push' in command:
                events.append(('push', command[-1]))
                return 'Kernel version 1 successfully pushed.'
            if 'status' in command:
                return f'example has status "KernelWorkerStatus.{statuses.pop(0)}"'
            if 'output' in command:
                events.append(('download', command[-1]))
                return ''
            raise AssertionError(command)
        def verify(archive):
            value = verifications.pop(0)
            if isinstance(value, Exception):
                raise value
            events.append(('verify', value['epochs_completed']))
            return value
        return prepare, execute, verify, events

    def test_verified_jobs_continue_25_then_24_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline = workspace / 'outputs/kaggle_results/session_03/verified/unetpp_rd_002_amp'
            baseline.mkdir(parents=True)
            metrics = {'sensitivity': .8, 'specificity': .9, 'average_precision': .7,
                       'accuracy': .9, 'FN': 8, 'FP': 10}
            (baseline / 'summary.json').write_text(json.dumps({'best_validation_metrics': metrics}))
            reports = [{'passed': True, 'epochs_completed': n, 'last_epoch_seconds': 488,
                        'selected_validation_metrics': deepcopy(metrics)} for n in [26, 50]]
            prepare, execute, verify, events = self.setup_fakes(tmp, ['RUNNING', 'COMPLETE', 'COMPLETE'], reports)
            state = initial_state()
            with patch.object(controller, 'ROOT', workspace), patch.object(controller, 'config_hash', return_value='fixed'), patch.object(controller, 'prepare', prepare):
                controller.drive(state, workspace, execute, verify, sleep=lambda _: None)
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['epochs_completed'], 50)
            planned = [e for e in events if e[0] == 'prepare']
            self.assertEqual([e[2] for e in planned], [25, 24])
            self.assertEqual(planned[1][3], 'example/unetpp-rd-aug-001-train-02/1')
            self.assertLess(events.index(('verify', 26)), events.index(planned[1]))
            self.assertTrue((workspace / 'validation_comparison.csv').is_file())

    def test_remote_error_never_submits_a_later_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepare, execute, verify, events = self.setup_fakes(tmp, ['ERROR'], [])
            with patch.object(controller, 'config_hash', return_value='fixed'), patch.object(controller, 'prepare', prepare):
                with self.assertRaisesRegex(RuntimeError, 'Remote job ended with ERROR'):
                    controller.drive(initial_state(), Path(tmp), execute, verify, sleep=lambda _: None)
            self.assertEqual(len([e for e in events if e[0] == 'push']), 1)

    def test_failed_verification_never_submits_a_later_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepare, execute, verify, events = self.setup_fakes(tmp, ['COMPLETE'], [ValueError('corrupt archive')])
            with patch.object(controller, 'config_hash', return_value='fixed'), patch.object(controller, 'prepare', prepare):
                with self.assertRaisesRegex(ValueError, 'corrupt archive'):
                    controller.drive(initial_state(), Path(tmp), execute, verify, sleep=lambda _: None)
            self.assertEqual(len([e for e in events if e[0] == 'push']), 1)

    def test_uncertain_submission_is_not_repeated_on_restart(self):
        state = initial_state()
        state['current_job'] = {'phase': 'submitting'}
        with tempfile.TemporaryDirectory() as tmp, patch.object(controller, 'config_hash', return_value='fixed'):
            calls = []
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                controller.drive(state, Path(tmp), execute=lambda c: calls.append(c))
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
