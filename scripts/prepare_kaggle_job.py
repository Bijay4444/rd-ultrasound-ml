#!/usr/bin/env python3
"""Create private Kaggle submission bundles; never submits or starts jobs itself."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from kaggle_cli import credential_environment


def write_bundle(username, kind, number, output_root, epochs=1, previous=None):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{2,29}', username):
        raise ValueError('Enter your public Kaggle username, not a URL or email')
    username = username.lower()
    if kind not in {'prepare', 'train'} or (kind == 'prepare' and number not in {0, 1}):
        raise ValueError('Invalid job type or preparation shard')
    if kind == 'train' and (number < 1 or not 1 <= epochs <= 50):
        raise ValueError('Training session and epoch count must be positive (at most 50 epochs)')
    if previous is not None and not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+', previous):
        raise ValueError('Previous source must have the form username/notebook-slug')
    if kind == 'train' and number > 1:
        previous = previous or f'{username}/unetpp-rd-002-train-{number-1:02d}'
    if kind == 'train' and number == 1 and previous:
        raise ValueError('Use session 2 or later when attaching a preceding training run')
    filename = 'kaggle_unetpp_preparation.ipynb' if kind == 'prepare' else 'kaggle_unetpp_training.ipynb'
    notebook = json.loads((ROOT/'notebooks'/filename).read_text())
    settings = next(cell for cell in notebook['cells'] if cell['id'] == 'kaggle-settings')
    if kind == 'prepare':
        settings['source'] = [f'UNETPP_SHARD_INDEX = {number}\n']
        slug = f'unetpp-rd-002-inputs-{number:02d}'
        title = f'UNetpp RD 002 Inputs {number:02d}'
        dependencies = []
    else:
        settings['source'] = [f'UNETPP_EPOCHS_THIS_SESSION = {epochs}\n',
            f'UNETPP_REQUIRE_PREVIOUS = {previous is not None}\n', 'UNETPP_MAX_SESSION_SECONDS = 6 * 3600\n']
        slug = f'unetpp-rd-002-train-{number:02d}'
        title = f'UNetpp RD 002 Train {number:02d}'
        dependencies = [f'{username}/unetpp-rd-002-inputs-{index:02d}' for index in range(2)]
        if previous:
            dependencies.append(previous)
    metadata = {'id': f'{username}/{slug}', 'title': title, 'code_file': filename,
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': kind == 'train', 'enable_internet': kind == 'prepare',
        'dataset_sources': [], 'competition_sources': [], 'kernel_sources': dependencies}
    if kind == 'train':
        metadata['machine_shape'] = 'NvidiaTeslaT4'
    destination = Path(output_root)/slug
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in [(filename, notebook), ('kernel-metadata.json', metadata)]:
        path = destination/name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f'Existing bundle differs: {path}. Keep it for audit; choose a separate output root.')
        path.write_text(json.dumps(value, indent=1)+'\n')
    return destination, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['init', 'train'])
    parser.add_argument('--username', default=credential_environment().get('KAGGLE_USERNAME'))
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/kaggle_jobs')
    parser.add_argument('--session', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--previous', help='Preceding successfully saved GPU notebook, owner/slug')
    args = parser.parse_args()
    if not args.username:
        parser.error('Set --username or run bash scripts/setup_kaggle.sh first')
    jobs = [('prepare', 0), ('prepare', 1), ('train', 1)] if args.stage == 'init' else [('train', args.session)]
    for kind, number in jobs:
        destination, metadata = write_bundle(args.username, kind, number, args.output_root, args.epochs, args.previous)
        print(f'{metadata["id"]}: {destination}')
        print(f'  Submit: python3 scripts/kaggle_cli.py kernels push -p {destination}')
        print(f'  Status: python3 scripts/kaggle_cli.py kernels status {metadata["id"]}')
    print('Submit both CPU jobs first. Wait for complete status and saved outputs before submitting the GPU job.')


if __name__ == '__main__':
    main()
