#!/usr/bin/env python3
"""Create private, version-pinned test-preparation or evaluation submission bundles."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from kaggle_cli import credential_environment


def write_evaluation_bundle(stage, username, output_root):
    freeze=json.loads((ROOT/'configs/unetpp_rd_002_evaluation.json').read_text())
    owner=freeze['training_source'].split('/')[0]
    if username != owner:
        raise ValueError('Evaluation must use the owner of the frozen private training artifacts')
    if stage not in {'prepare','evaluate'}:
        raise ValueError('Choose prepare or evaluate')
    preparation=f'{owner}/unetpp-rd-002-test-inputs'
    filename='kaggle_unetpp_test_preparation.ipynb' if stage=='prepare' else 'kaggle_unetpp_evaluation.ipynb'
    slug='unetpp-rd-002-test-inputs' if stage=='prepare' else 'unetpp-rd-002-test-evaluation'
    sources=[freeze['training_source']+'/'+str(freeze['training_source_version'])]
    if stage=='evaluate':sources.append(preparation+'/1')
    metadata={'id':owner+'/'+slug,'title':'UNetpp RD 002 Test Inputs' if stage=='prepare' else 'UNetpp RD 002 Test Evaluation',
        'code_file':filename,'language':'python','kernel_type':'notebook','is_private':True,
        'enable_gpu':stage=='evaluate','enable_internet':stage=='prepare',
        'dataset_sources':[],'competition_sources':[],'kernel_sources':sources}
    if stage=='evaluate':
        metadata['machine_shape']='NvidiaTeslaT4'
        runtime_pin_path=ROOT/'configs/unetpp_rd_002_runtime.json'
        if runtime_pin_path.exists():
            runtime_pin=json.loads(runtime_pin_path.read_text())
            if (runtime_pin['training_source'] != freeze['training_source']
                    or runtime_pin['training_source_version'] != freeze['training_source_version']
                    or not re.fullmatch(r'gcr\.io/[^\s]+@sha256:[0-9a-f]{64}',runtime_pin['docker_image'])):
                raise ValueError('Runtime pin does not identify the frozen training source/container')
            metadata['docker_image']=runtime_pin['docker_image']
            metadata['docker_image_pinning_type']='original'
    directory=Path(output_root)/slug;directory.mkdir(parents=True,exist_ok=True)
    notebook=json.loads((ROOT/'notebooks'/filename).read_text())
    for name,value in [(filename,notebook),('kernel-metadata.json',metadata)]:
        path=directory/name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Existing evaluation bundle differs; preserve it before a documented revision')
        path.write_text(json.dumps(value,indent=1)+'\n')
    return directory,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','evaluate'])
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/kaggle_jobs')
    args=parser.parse_args()
    username=credential_environment().get('KAGGLE_USERNAME','').lower()
    if not username:parser.error('Run the Kaggle setup wizard to record the account username first')
    directory,metadata=write_evaluation_bundle(args.stage,username,args.output_root)
    print(metadata['id'],directory)
    print(f'Submit: python3 scripts/kaggle_cli.py kernels push -p {directory}')
    print(f'Status: python3 scripts/kaggle_cli.py kernels status {metadata["id"]}')
    if args.stage=='evaluate':print('Submit only once CPU test preparation version 1 has saved successfully.')


if __name__=='__main__':main()
