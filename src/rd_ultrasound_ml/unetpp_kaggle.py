"""Kaggle CPU-output shards and verified input/checkpoint handoff.

Platform/storage adaptation only: arrays retain the original float32 values.
"""
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd

from rd_ultrasound_ml.unetpp_data import verify_partitions
from rd_ultrasound_ml.unetpp_training import CheckpointStore, file_digest, json_digest

METADATA_MD5 = '95828fb5bdffdbf8173b5bcfc99e8148'
SPLITS_MD5 = 'c5630e3f056ce358dcd384aaed946e64'


def shard_indices(length, index, count=2):
    if length < count or count < 1 or not 0 <= index < count:
        raise ValueError('Invalid preparation shard dimensions')
    return np.array_split(np.arange(length), count)[index].tolist()


def read_official_inputs(directory):
    import hashlib
    directory = Path(directory)
    metadata = (directory / 'erdes_metadata.csv').read_bytes()
    splits = (directory / 'splits.zip').read_bytes()
    if hashlib.md5(metadata).hexdigest() != METADATA_MD5 or hashlib.md5(splits).hexdigest() != SPLITS_MD5:
        raise ValueError('Official metadata/split archive checksum differs')
    with zipfile.ZipFile(io.BytesIO(splits)) as archive:
        train = pd.read_csv(io.BytesIO(archive.read('splits/non_rd_vs_rd/train.csv')))
        validation = pd.read_csv(io.BytesIO(archive.read('splits/non_rd_vs_rd/val.csv')))
    return verify_partitions(train, validation, pd.read_csv(io.BytesIO(metadata)))


def load_prepared_shards(input_root, config, report=print):
    """Reject incomplete/duplicate shards and verify every attached tensor before use."""
    input_root = Path(input_root)
    manifests = sorted(input_root.rglob('prepared_manifest.json'))
    if len(manifests) != 2:
        raise ValueError(f'Attach exactly two completed CPU preparation outputs; found {len(manifests)} manifests')
    train, validation = read_official_inputs(manifests[0].parent)
    all_rows = pd.concat([train, validation], ignore_index=True)
    records, paths, seen_shards = {}, {}, set()
    identity = None
    for path in manifests:
        manifest = json.loads(path.read_text())
        if manifest.get('complete') is not True or manifest.get('shard_count') != 2:
            raise ValueError('Preparation output is incomplete or has the wrong shard count')
        index = manifest['shard_index']
        if index in seen_shards or index not in {0, 1}:
            raise ValueError('Duplicate or invalid preparation shard')
        seen_shards.add(index)
        if manifest['config_digest'] != json_digest(config):
            raise ValueError('CPU preparation configuration/code differs from this training notebook')
        if identity is None:
            identity = manifest['preprocessing_identity']
        elif identity != manifest['preprocessing_identity']:
            raise ValueError('CPU shard preprocessing versions differ')
        expected = shard_indices(len(all_rows), index)
        entries = manifest['entries']
        if [entry['global_index'] for entry in entries] != expected:
            raise ValueError('Preparation shard omitted, duplicated or reordered selected clips')
        other_train, other_val = read_official_inputs(path.parent)
        if not train.equals(other_train) or not validation.equals(other_val):
            raise ValueError('CPU shards contain different official partitions')
        for entry in entries:
            position = entry['global_index']
            row = all_rows.iloc[position]
            check = entry['check']
            if entry['clip_id'] != row['clip_id'] or entry['label'] != int(row['label']):
                raise ValueError('Prepared clip identity/label differs from official manifest')
            expected_identity = {'signature': json_digest(identity), 'clip_id': row['clip_id'],
                'file_path': row['file_path'], 'source_sha256': entry['source']['sha256'],
                'frame_count': int(row['frame_count']), 'height': int(row['height']),
                'width': int(row['width']), 'target': [96, 128, 128]}
            if check['identity'] != expected_identity:
                raise ValueError('Prepared clip provenance differs from selected source/preprocessing')
            relative = Path(entry['array'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe prepared-array path')
            array_path = path.parent / relative
            if file_digest(array_path) != check['tensor_sha256']:
                raise ValueError(f'Prepared tensor checksum differs: {row["clip_id"]}')
            array = np.load(array_path, mmap_mode='r', allow_pickle=False)
            if array.dtype != np.float32 or array.shape != (1, 96, 128, 128):
                raise ValueError('Prepared tensor shape/dtype differs from fixed protocol')
            paths[position], records[position] = array_path, entry
            if len(paths) % 100 == 0:
                report(f'Verified attached input {len(paths)}/{len(all_rows)}')
    if set(paths) != set(range(len(all_rows))):
        raise ValueError('Prepared inputs do not cover the complete official train/validation sets')
    ordered_records = [records[index] for index in range(len(all_rows))]
    fingerprint = json_digest({'config_digest': json_digest(config), 'preprocessing_identity': identity,
        'records': ordered_records,
        'train_manifest': train.to_csv(index=False), 'validation_manifest': validation.to_csv(index=False)})
    return train, validation, [paths[index] for index in range(len(all_rows))], fingerprint, identity


def import_previous_run(input_root, destination, config, previous_required=False):
    """Copy only saved training artifacts; never write to attached read-only inputs."""
    destination = Path(destination)
    candidates = sorted(Path(input_root).rglob('last_checkpoint.json'))
    if (destination / 'last_checkpoint.json').exists():
        return destination  # Current interactive session already owns the latest state.
    if len(candidates) > 1:
        raise ValueError('Attach only one preceding training output to avoid ambiguous resume')
    if not candidates:
        if previous_required:
            raise ValueError('This continuation requires the preceding successful GPU notebook output')
        return None
    source = candidates[0].parent
    previous_config = json.loads((source / 'config.json').read_text())
    if previous_config != config:
        raise ValueError('Preceding training configuration differs; do not initialize over it')
    destination.mkdir(parents=True, exist_ok=True)
    # Include the per-epoch prediction records needed to recover the selected best epoch.
    for path in source.iterdir():
        if path.is_file() and path.suffix in {'.json', '.csv', '.pt', '.py'}:
            shutil.copy2(path, destination / path.name)
    if CheckpointStore(destination).load() is None:
        raise ValueError('Preceding training output lacks a valid checkpoint')
    return source
