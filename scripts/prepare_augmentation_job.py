"""Prepare private, version-pinned augmentation jobs; never submits them."""
import argparse
import json
from pathlib import Path
import re

from build_augmentation_notebook import notebook, FILENAME

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ['bijaykhadka1710/unetpp-rd-002-inputs-00/2', 'bijaykhadka1710/unetpp-rd-002-inputs-01/2']
# Retrieved with `kernels pull .../unetpp-rd-002-train-03/1 -m`.
# New notebooks otherwise receive the latest environment, which has drifted.
DOCKER_IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435'


def prepare(username, output_root, session=1, epochs=1, previous=None):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{2,29}', username):
        raise ValueError('Supply a lowercase public Kaggle username')
    if type(session) is not int or session < 1 or type(epochs) is not int or not 1 <= epochs <= 50:
        raise ValueError('Expected positive session and 1-50 epochs')
    if session == 1 and previous:
        raise ValueError('Initial training must start from scratch without a previous checkpoint')
    if session > 1 and (not previous or not re.fullmatch(
            rf'{re.escape(username)}/unetpp-rd-aug-001-train-{session-1:02d}/[1-9][0-9]*', previous)):
        raise ValueError('Continuation requires the preceding augmentation output, owner/slug/version')
    generated = notebook()
    settings = next(c for c in generated['cells'] if c['id'] == 'kaggle-settings')
    settings['source'] = [f'AUG_EPOCHS_THIS_SESSION = {epochs}\n',
        f'AUG_REQUIRE_PREVIOUS = {previous is not None}\n', 'AUG_MAX_SESSION_SECONDS = 6 * 3600\n']
    slug = f'unetpp-rd-aug-001-train-{session:02d}'
    metadata = {'id': f'{username}/{slug}', 'title': f'UNetpp RD Aug 001 Train {session:02d}',
        'code_file': FILENAME, 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': True, 'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4',
        'docker_image': DOCKER_IMAGE,
        'dataset_sources': [], 'competition_sources': [], 'kernel_sources': INPUTS + ([previous] if previous else [])}
    output = Path(output_root) / slug
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [(FILENAME, generated), ('kernel-metadata.json', metadata)]:
        target = output / name
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError('Existing bundle differs; preserve it and choose a fresh output root')
    for name, value in [(FILENAME, generated), ('kernel-metadata.json', metadata)]:
        (output / name).write_text(json.dumps(value, indent=1) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username', required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/kaggle_jobs')
    parser.add_argument('--session', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--previous')
    args = parser.parse_args()
    print(prepare(args.username, args.output_root, args.session, args.epochs, args.previous))
