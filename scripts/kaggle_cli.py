#!/usr/bin/env python3
"""Run Kaggle's isolated CLI, reading only approved keys from ignored local dotenv."""
import os
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def credential_environment(path=ROOT / '.env.kaggle', base=None):
    environment = dict(os.environ if base is None else base)
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            key, separator, value = line.partition('=')
            if not separator or key not in {'KAGGLE_USERNAME', 'KAGGLE_API_TOKEN'}:
                raise ValueError('Unexpected setting in .env.kaggle')
            parsed = shlex.split(value, comments=False, posix=True)
            if len(parsed) != 1 or not parsed[0]:
                raise ValueError('Empty or invalid Kaggle setting')
            environment.setdefault(key, parsed[0])
    return environment


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python3 scripts/kaggle_cli.py --help')
    # six is absent from the CLI 2.2.5 dependency metadata but imported by its upload helper.
    cli = ['uv', '--cache-dir', str(ROOT/'outputs/.kaggle-uv-cache'), 'run', '--no-project',
           '--with', 'kaggle==2.2.5', '--with', 'six', 'kaggle']
    arguments, environment = sys.argv[1:], credential_environment()
    if arguments[:2] == ['kernels', 'push'] and '--no-run' not in arguments:
        folder = ROOT
        for index, argument in enumerate(arguments):
            if argument in {'-p', '--path'}:
                folder = Path(arguments[index + 1])
            elif argument.startswith('--path='):
                folder = Path(argument.split('=', 1)[1])
        metadata_path = folder/'kernel-metadata.json'
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if str(metadata.get('enable_gpu', False)).lower() == 'true':
                dependencies = metadata.get('kernel_sources', [])
                if not dependencies:
                    print('Not submitted: GPU training needs its completed preparation outputs.', file=sys.stderr)
                    return 1
                for source in dependencies:
                    result = subprocess.run(cli + ['kernels', 'status', source], cwd=ROOT,
                        env=environment, capture_output=True, text=True, timeout=90)
                    if result.returncode or not re.search(
                            r'has status ["\'](?:KernelWorkerStatus\.)?COMPLETE["\']', result.stdout, re.IGNORECASE):
                        print(f'Not submitted: required input {source} is not successfully complete. '
                              'Finish CPU preparation (and the preceding GPU job for continuation) first.', file=sys.stderr)
                        print(result.stdout.strip() or 'Could not verify its remote status.', file=sys.stderr)
                        return 1
                    print(f'Input complete: {source}', flush=True)
    return subprocess.call(cli + arguments, cwd=ROOT, env=environment)


if __name__ == '__main__':
    sys.exit(main())
