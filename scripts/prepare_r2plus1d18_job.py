#!/usr/bin/env python3
"""Create private Kaggle GPU bundles for r2plus1d18_rd_001_amp; never submits jobs itself.

Inputs are the two completed U-Net++ CPU preparation outputs, given explicitly as
owner/slug/version (version-pinned). Bijay must confirm and share those exact
sources; this script does not guess them. Continuations attach the preceding
successful R(2+1)D GPU output as well.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from kaggle_cli import credential_environment

FILENAME = 'kaggle_r2plus1d18_training.ipynb'
SLUG = 'r2plus1d18-rd-001-train-{:02d}'
SOURCE = re.compile(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+(/[1-9][0-9]*)?')


def write_bundle(username, session, inputs, output_root, epochs=1, previous=None):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{2,29}', username):
        raise ValueError('Enter your public Kaggle username, not a URL or email')
    username = username.lower()
    if session < 1 or not 1 <= epochs <= 50:
        raise ValueError('Session must be positive and epochs 1-50')
    if len(inputs) != 2 or len(set(inputs)) != 2:
        raise ValueError('Give exactly two distinct preparation outputs (shards 00 and 01)')
    for source in inputs:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/[1-9][0-9]*', source):
            raise ValueError('Preparation inputs must be version-pinned: owner/slug/version')
    if session > 1:
        previous = previous or f'{username}/{SLUG.format(session - 1)}'
    elif previous:
        raise ValueError('Use session 2 or later when attaching a preceding training run')
    if previous is not None and not SOURCE.fullmatch(previous):
        raise ValueError('Previous source must have the form owner/slug or owner/slug/version')
    notebook = json.loads((ROOT/'notebooks'/FILENAME).read_text(encoding='utf-8'))
    settings = next(cell for cell in notebook['cells'] if cell['id'] == 'kaggle-settings')
    settings['source'] = [f'R2P1_EPOCHS_THIS_SESSION = {epochs}\n',
        f'R2P1_REQUIRE_PREVIOUS = {previous is not None}\n', 'R2P1_MAX_SESSION_SECONDS = 11 * 3600\n']
    slug = SLUG.format(session)
    metadata = {'id': f'{username}/{slug}', 'title': slug.replace('-', ' ').title(),
        'code_file': FILENAME, 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': True, 'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4',
        'dataset_sources': [], 'competition_sources': [],
        'kernel_sources': [*inputs, *([previous] if previous else [])]}
    destination = Path(output_root)/slug
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in [(FILENAME, notebook), ('kernel-metadata.json', metadata)]:
        path = destination/name
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'Existing bundle differs: {path}. Keep it for audit; choose a separate output root.')
        path.write_text(json.dumps(value, indent=1) + '\n', encoding='utf-8')
    return destination, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--username', default=credential_environment().get('KAGGLE_USERNAME'))
    parser.add_argument('--inputs', nargs=2, required=True, metavar='OWNER/SLUG/VERSION',
                        help='The two completed U-Net++ preparation outputs, shard 00 then 01')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/kaggle_jobs')
    parser.add_argument('--session', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--previous', help='Preceding successfully saved R(2+1)D GPU notebook')
    args = parser.parse_args()
    if not args.username:
        parser.error('Set --username or run bash scripts/setup_kaggle.sh first')
    destination, metadata = write_bundle(args.username, args.session, args.inputs, args.output_root,
                                         args.epochs, args.previous)
    print(f'{metadata["id"]}: {destination}')
    print(f'  Submit: python3 scripts/kaggle_cli.py kernels push -p {destination}')
    print(f'  Status: python3 scripts/kaggle_cli.py kernels status {metadata["id"]}')


if __name__ == '__main__':
    main()
