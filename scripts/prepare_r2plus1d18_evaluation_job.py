#!/usr/bin/env python3
"""Create the private, version-pinned R(2+1)D-18 test-evaluation bundle (does not submit).

Refuses unless the tracked evaluation config has team_freeze_confirmed: true.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from kaggle_cli import credential_environment

FREEZE = ROOT/'configs/r2plus1d18_rd_001_evaluation.json'
FILENAME = 'kaggle_r2plus1d18_evaluation.ipynb'
SLUG = 'r2plus1d18-rd-001-test-evaluation'


def write_evaluation_bundle(username, output_root, freeze=None):
    freeze = freeze or json.loads(FREEZE.read_text(encoding='utf-8'))
    if freeze.get('team_freeze_confirmed') is not True:
        raise ValueError('Team freeze not confirmed: set team_freeze_confirmed in '
                         'configs/r2plus1d18_rd_001_evaluation.json in a reviewed commit first')
    owner = freeze['training_source'].split('/')[0]
    if username.lower() != owner:
        raise ValueError('Evaluation must run under the owner of the frozen private training output')
    sources = [f"{freeze['training_source']}/{freeze['training_source_version']}",
               f"{freeze['test_cache']['source']}/{freeze['test_cache']['source_version']}"]
    metadata = {'id': f'{owner}/{SLUG}', 'title': 'R2Plus1D18 Rd 001 Test Evaluation', 'code_file': FILENAME,
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True,
        'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4',
        'dataset_sources': [], 'competition_sources': [], 'kernel_sources': sources}
    notebook = json.loads((ROOT/'notebooks'/FILENAME).read_text(encoding='utf-8'))
    directory = Path(output_root)/SLUG
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in [(FILENAME, notebook), ('kernel-metadata.json', metadata)]:
        path = directory/name
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Existing evaluation bundle differs; preserve it before a documented revision')
        path.write_text(json.dumps(value, indent=1) + '\n', encoding='utf-8')
    return directory, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username', default=credential_environment().get('KAGGLE_USERNAME', ''))
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/kaggle_jobs')
    args = parser.parse_args()
    if not args.username:
        parser.error('Run the Kaggle setup wizard to record the account username first')
    directory, metadata = write_evaluation_bundle(args.username, args.output_root)
    print(metadata['id'], directory)
    print(f'Submit (once, after team approval): python3 scripts/kaggle_cli.py kernels push -p {directory}')


if __name__ == '__main__':
    main()
