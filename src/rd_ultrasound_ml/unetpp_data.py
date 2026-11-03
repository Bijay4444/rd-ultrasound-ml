"""Verified full-video cache for the fixed UNet++ train/validation protocol."""
import json
from pathlib import Path
import time
from urllib.parse import quote

import numpy as np
import torch
from torch.utils.data import Dataset

from rd_ultrasound_ml.unetpp_baseline import prepare_unetpp_clip
from rd_ultrasound_ml.unetpp_training import file_digest, json_digest, write_json_atomic


def verify_partitions(train, validation, metadata):
    """Preserve supplied CSV order and reject dropped rows, overlap or label disagreement."""
    merged = []
    for frame, count, labels in [(train, 3873, {0: 3511, 1: 362}),
                                  (validation, 431, {0: 391, 1: 40})]:
        if len(frame) != count or not frame['path'].is_unique or frame['label'].value_counts().to_dict() != labels:
            raise ValueError('Official partition size, uniqueness or label counts differ')
        joined = frame.merge(metadata, left_on='path', right_on='file_path', how='left',
                             sort=False, validate='one_to_one')
        if joined['clip_id'].isna().any() or not joined['label'].map({0: 'non_rd', 1: 'rd'}).eq(joined['diagnostic_class']).all():
            raise ValueError('Partition/metadata labels or paths disagree')
        if not joined['clip_id'].is_unique or not joined['path'].tolist() == frame['path'].tolist():
            raise ValueError('Manifest identity/order changed')
        for path in joined['file_path']:
            if Path(path).is_absolute() or '..' in Path(path).parts:
                raise ValueError('Unsafe relative clip path')
        merged.append(joined)
    if set(train['path']) & set(validation['path']):
        raise ValueError('Training and validation clips overlap')
    return tuple(merged)


def record_video_sources(frame, revision, path, api, report=print):
    """Persist each verified API chunk, using immutable-revision LFS hashes."""
    path = Path(path)
    catalog = json.loads(path.read_text()) if path.exists() else {
        'dataset': 'pcvlab/erdes', 'revision': revision,
        'license_declaration': 'cc-by-4.0 (author dataset card)', 'files': {}}
    if catalog['revision'] != revision or catalog['dataset'] != 'pcvlab/erdes':
        raise ValueError('Source catalog revision differs')
    files = catalog['files']
    missing = [value for value in frame['file_path'] if value not in files]
    for start in range(0, len(missing), 50):
        requested = ['clips/' + value for value in missing[start:start + 50]]
        for attempt in range(5):
            try:
                entries = api.get_paths_info('pcvlab/erdes', paths=requested, repo_type='dataset',
                                             revision=revision, token=False)
                break
            except Exception as error:
                response = getattr(error, 'response', None)
                status = getattr(response, 'status_code', None)
                if status not in {408, 429, 500, 502, 503, 504} or attempt == 4:
                    raise
                retry_after = response.headers.get('Retry-After', '')
                delay = max(2 ** (attempt + 1), float(retry_after) if retry_after.isdigit() else 0)
                if delay > 60:
                    raise RuntimeError(f'Source API requests a wait of {delay}s; wait, then rerun preparation') from None
                report(f'Source catalog HTTP {status}; retry in {delay}s')
                time.sleep(delay)
        by_path = {entry.path: entry for entry in entries}
        if set(by_path) != set(requested):
            raise ValueError('Source API omitted selected clips')
        for requested_path in requested:
            entry = by_path[requested_path]
            if entry.lfs is None or len(entry.lfs.sha256) != 64 or entry.size != entry.lfs.size:
                raise ValueError(f'Missing source SHA256/size: {requested_path}')
            files[requested_path.removeprefix('clips/')] = {
                'sha256': entry.lfs.sha256, 'bytes': int(entry.size),
                'url': f'https://huggingface.co/datasets/pcvlab/erdes/resolve/{revision}/{quote(requested_path, safe="/")}'}
        write_json_atomic(path, catalog)
        report(f'Source hashes recorded: {len(files)} clips')
    for value in frame['file_path']:
        source = files[value]
        expected_url = f'https://huggingface.co/datasets/pcvlab/erdes/resolve/{revision}/clips/{quote(value, safe="/")}'
        if (len(source['sha256']) != 64 or any(c not in '0123456789abcdef' for c in source['sha256'])
                or source['bytes'] < 1 or source['url'] != expected_url):
            raise ValueError(f'Invalid saved source provenance: {value}')
    return files


def prepare_cached_clip(row, source, data_dir, cache_dir, cache_signature, download, target=(96, 128, 128)):
    """Commit a tensor only after decode/range checks; validate hashes on every reuse."""
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = {'signature': cache_signature, 'clip_id': row.clip_id, 'file_path': row.file_path,
        'source_sha256': source['sha256'], 'frame_count': int(row.frame_count),
        'height': int(row.height), 'width': int(row.width), 'target': list(target)}
    name = json_digest(identity)
    tensor_path, check_path = cache_dir / (name + '.npy'), cache_dir / (name + '.json')
    if check_path.exists():
        check = json.loads(check_path.read_text())
        if check['identity'] != identity or not tensor_path.is_file() or file_digest(tensor_path) != check['tensor_sha256']:
            raise ValueError(f'Cached tensor integrity failure: {row.clip_id}; retain the file for inspection')
        tensor = np.load(tensor_path, mmap_mode='r', allow_pickle=False)
        if tensor.shape != (1, *target) or tensor.dtype != np.float32:
            raise ValueError(f'Cached tensor shape/dtype failure: {row.clip_id}')
        return tensor_path, check
    video_path = data_dir / 'clips' / row.file_path
    download(source['url'], video_path, source['sha256'], expected_bytes=source['bytes'])
    tensor = prepare_unetpp_clip(video_path, int(row.frame_count), int(row.height), int(row.width), target=target)
    temporary = tensor_path.with_suffix('.npy.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, tensor.numpy(), allow_pickle=False)
    check = {'identity': identity, 'shape': list(tensor.shape), 'dtype': 'float32',
        'min': float(tensor.min()), 'max': float(tensor.max()), 'tensor_sha256': file_digest(temporary)}
    temporary.replace(tensor_path)
    write_json_atomic(check_path, check)
    return tensor_path, check


class CachedClips(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = list(map(Path, paths)), np.asarray(labels, dtype=np.float32)
        if len(self.paths) == 0 or len(self.paths) != len(self.labels) or not np.isin(self.labels, [0, 1]).all():
            raise ValueError('Expected matching non-empty cached paths and binary labels')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # One file at a time; no 27 GB concatenation or retained memmap handles.
        array = np.load(self.paths[index], allow_pickle=False)
        return torch.from_numpy(array), torch.tensor(self.labels[index])
