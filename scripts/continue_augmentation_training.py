"""Continue the fixed 50-epoch experiment through bounded, verified private jobs.

Runs locally; keep this process/WSL alive for automatic handoffs. An already
submitted Kaggle job continues independently. Failures stop future submissions;
they never delete remote runs or restart training from fresh weights.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from prepare_augmentation_job import prepare

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = 'unetpp_rd_aug_001_amp_results.zip'


def config_hash():
    return hashlib.sha256((ROOT / 'configs/unetpp_rd_aug_001_training.json').read_bytes()).hexdigest()


def save(state, workspace):
    state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    workspace.mkdir(parents=True, exist_ok=True)
    temporary = workspace / 'state.json.tmp'
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    temporary.replace(workspace / 'state.json')


def run_command(arguments):
    result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-3000:])
    return result.stdout


def kaggle(*arguments):
    return [sys.executable, str(ROOT / 'scripts/kaggle_cli.py'), *arguments]


def verify_output(archive):
    command = ['uv', '--cache-dir', str(ROOT / 'outputs/.pilot-uv-cache'), 'run', '--offline', '--no-project']
    for package in ('torch', 'torchvision', 'pandas', 'nbformat', 'opencv-python-headless',
                    'ipython', 'scikit-learn', 'huggingface-hub', 'matplotlib', 'pillow'):
        command += ['--with', package]
    command += ['--index', 'https://download.pytorch.org/whl/cpu', 'python',
        str(ROOT / 'scripts/verify_augmentation_results.py'), '--archive', str(archive)]
    run_command(command)
    return json.loads((archive.parent / 'verification.json').read_text())


def drive(state, workspace, execute=run_command, verify_saved=verify_output, sleep=time.sleep):
    """Persist each submission boundary so a restart cannot silently submit twice."""
    workspace = Path(workspace)
    while state['epochs_completed'] < 50:
        if state['config_sha256'] != config_hash():
            raise ValueError('Frozen config changed; stop rather than launching a different experiment')
        current = state.get('current_job')
        if current is None:
            session = state['last_session'] + 1
            # Keep at least 15 minutes for setup/export and a 50% timing margin.
            budget_count = int((6 * 3600 - 900) / (1.5 * state['last_epoch_seconds']))
            count = min(25, 50 - state['epochs_completed'], budget_count)
            if count < 1:
                raise RuntimeError('Measured epoch time cannot fit the bounded session budget')
            bundle = prepare(state['username'], workspace / 'jobs', session, count, state['previous_source'])
            metadata = json.loads((bundle / 'kernel-metadata.json').read_text())
            listing = execute(kaggle('kernels', 'list', '--mine', '--page-size', '100', '--csv'))
            existing = {row['ref'] for row in csv.DictReader(io.StringIO(listing))}
            if metadata['id'] in existing:
                raise RuntimeError('Next job already exists but is not recorded here; inspect before submitting')
            current = {'ref': metadata['id'], 'session': session, 'epochs_requested': count,
                'starting_epoch_count': state['epochs_completed'], 'bundle': str(bundle), 'phase': 'prepared'}
            state['current_job'] = current
            save(state, workspace)
        if current['phase'] == 'prepared':
            current['phase'] = 'submitting'
            save(state, workspace)
            output = execute(kaggle('kernels', 'push', '-p', current['bundle']))
            match = re.search(r'Kernel version (\d+) successfully pushed', output)
            if not match:
                raise RuntimeError('Submission response is uncertain; inspect remote state before retrying')
            current.update(version=int(match[1]), phase='submitted')
            save(state, workspace)
            print(f"Submitted {current['ref']}/{current['version']}: {current['epochs_requested']} epochs", flush=True)
        elif current['phase'] == 'submitting':
            raise RuntimeError('Previous submission was interrupted; inspect remote state before retrying')
        while current['phase'] == 'submitted':
            status = execute(kaggle('kernels', 'status', current['ref']))
            match = re.search(r'has status ["\'](?:KernelWorkerStatus\.)?([A-Z_]+)["\']', status, re.IGNORECASE)
            if not match:
                raise RuntimeError('Could not determine job status; no later job will be submitted')
            remote = match[1].upper()
            current['remote_status'] = remote
            save(state, workspace)
            print(f"{current['ref']}: {remote}", flush=True)
            if remote == 'COMPLETE':
                current['phase'] = 'downloading'
                save(state, workspace)
                break
            if remote not in {'RUNNING', 'QUEUED'}:
                raise RuntimeError(f'Remote job ended with {remote}; saved earlier checkpoints are retained')
            sleep(60)
        if current['phase'] == 'downloading':
            output = workspace / 'results' / f"session_{current['session']:02d}_v{current['version']}"
            output.mkdir(parents=True, exist_ok=True)
            execute(kaggle('kernels', 'output', f"{current['ref']}/{current['version']}",
                '-p', str(output), '--file-pattern', ARCHIVE))
            current.update(phase='verifying', archive=str(output / ARCHIVE))
            save(state, workspace)
        if current['phase'] != 'verifying':
            raise RuntimeError('Unknown saved continuation phase; inspect before proceeding')
        verified = verify_saved(Path(current['archive']))
        if (not verified['passed'] or not current['starting_epoch_count'] < verified['epochs_completed'] <=
                current['starting_epoch_count'] + current['epochs_requested']):
            raise ValueError('Saved epoch progress differs; no later job will be submitted')
        state.update(epochs_completed=verified['epochs_completed'], last_session=current['session'],
            previous_source=f"{current['ref']}/{current['version']}", last_epoch_seconds=verified['last_epoch_seconds'])
        state['completed_jobs'].append({**current, 'phase': 'verified', 'verification': verified})
        state['current_job'] = None
        save(state, workspace)
        print(f"Verified {state['epochs_completed']}/50 epochs", flush=True)
    final = state['completed_jobs'][-1]['verification']
    baseline = json.loads((ROOT / 'outputs/kaggle_results/session_03/verified/unetpp_rd_002_amp/summary.json').read_text())
    rows = [['metric', 'baseline_validation', 'augmentation_validation']]
    for key in ('sensitivity', 'specificity', 'average_precision', 'accuracy', 'FN', 'FP'):
        rows.append([key, baseline['best_validation_metrics'][key], final['selected_validation_metrics'][key]])
    with (workspace / 'validation_comparison.csv').open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    state['status'] = 'completed'
    save(state, workspace)
    print('All 50 epochs completed and verified. Validation comparison saved; no test evaluation performed.', flush=True)


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=ROOT / 'outputs/augmentation/training_pipeline')
    parser.add_argument('--initial-verification', type=Path, required=True)
    parser.add_argument('--username', default='bijaykhadka1710')
    parser.add_argument('--previous', default='bijaykhadka1710/unetpp-rd-aug-001-train-01/2')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = workspace / 'state.json'
        if path.exists():
            state = json.loads(path.read_text())
            if state['username'] != args.username:
                raise ValueError('Saved continuation belongs to another account')
        else:
            first = json.loads(args.initial_verification.read_text())
            if not first['passed'] or first['epochs_completed'] != 1:
                raise ValueError('Initialize only from the verified first epoch')
            state = {'status': 'running', 'config_sha256': config_hash(), 'username': args.username,
                'epochs_completed': 1, 'last_session': 1, 'previous_source': args.previous,
                'last_epoch_seconds': first['last_epoch_seconds'], 'current_job': None,
                'completed_jobs': [], 'test_used': False}
        try:
            state['status'] = 'running'
            save(state, workspace)
            drive(state, workspace)
        except Exception as error:
            state.update(status='stopped_on_error', error=f'{type(error).__name__}: {error}')
            save(state, workspace)
            raise


if __name__ == '__main__':
    main()
