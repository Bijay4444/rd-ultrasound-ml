"""Prepare a private, source-version-pinned Kaggle bundle; does not submit it."""
import argparse
import json
from pathlib import Path

from build_explainability_notebook import notebook

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'rd-gradcam-validation-001'


def prepare(username, output_root):
    if not username or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in username):
        raise ValueError('Supply a lowercase Kaggle username')
    output = Path(output_root)/SLUG
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'id': f'{username}/{SLUG}', 'title': 'RD GradCAM Validation 001',
        'code_file': 'kaggle_video_explainability.ipynb', 'language': 'python',
        'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True,
        'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4',
        'dataset_sources': [], 'competition_sources': [], 'kernel_sources': [
            'bijaykhadka1710/unetpp-rd-002-inputs-00/2',
            'bijaykhadka1710/unetpp-rd-002-inputs-01/2',
            'bijaykhadka1710/unetpp-rd-002-train-03/1',
            'namanbabariya/r2plus1d18-rd-001-train-04/1']}
    files = {'kernel-metadata.json': metadata, 'kaggle_video_explainability.ipynb': notebook()}
    for name, content in files.items():
        target = output/name
        if target.exists() and json.loads(target.read_text()) != content:
            raise ValueError('Existing bundle differs; choose a fresh --output-root')
    for name, content in files.items():
        (output/name).write_text(json.dumps(content, indent=1)+'\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username', required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/kaggle_jobs')
    args = parser.parse_args()
    print(prepare(args.username, args.output_root))
