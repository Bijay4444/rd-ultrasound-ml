"""Prepare a private, version-pinned R3D-only explanation job; does not submit."""
import argparse
import json
from pathlib import Path

from build_r3d18_explainability_notebook import notebook, NOTEBOOK

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'rd-gradcam-r3d18-validation-001'
SOURCES = ['bijaykhadka1710/unetpp-rd-002-inputs-00/2',
           'bijaykhadka1710/unetpp-rd-002-inputs-01/2',
           'bijaykhadka1710/unetpp-rd-002-train-03/1',
           'namanbabariya/r3d18-rd-001-train-03/1']


def prepare(username, output_root, docker_image):
    if not username or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in username):
        raise ValueError('Supply a lowercase Kaggle username')
    if '@sha256:' not in docker_image:
        raise ValueError('Pin the historical explanation runtime by Docker digest')
    output = Path(output_root)/SLUG
    metadata = {'id': username+'/'+SLUG, 'title': 'RD GradCAM R3D18 Validation 001',
        'code_file': NOTEBOOK.name, 'language':'python', 'kernel_type':'notebook',
        'is_private':True, 'enable_gpu':True, 'enable_internet':False,
        'machine_shape':'NvidiaTeslaT4', 'docker_image':docker_image,
        'dataset_sources':[], 'competition_sources':[], 'kernel_sources':SOURCES}
    files = {'kernel-metadata.json':metadata, NOTEBOOK.name:notebook()}
    for name, content in files.items():
        path = output/name
        if path.exists() and json.loads(path.read_text()) != content:
            raise ValueError('Existing bundle differs; choose a fresh output root')
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (output/name).write_text(json.dumps(content,indent=1)+'\n')
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username',required=True)
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/kaggle_jobs')
    parser.add_argument('--docker-image',required=True)
    args=parser.parse_args()
    print(prepare(args.username,args.output_root,args.docker_image))
