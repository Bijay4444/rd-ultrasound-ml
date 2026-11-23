"""Build real training-only augmentation previews from the eight inspected videos.

No downloads, training, validation/test inference, or cache modification.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def build(data_root, output, policy=None):
    import gc
    import cv2
    import numpy as np
    import torch
    from rd_ultrasound_ml.augmentation_preview import save_preview
    from rd_ultrasound_ml.unetpp_baseline import prepare_unetpp_clip
    from rd_ultrasound_ml.unetpp_data import CachedClips
    from rd_ultrasound_ml.unetpp_kaggle import read_official_inputs
    from rd_ultrasound_ml.unetpp_training import file_digest, write_json_atomic
    from rd_ultrasound_ml.video_augmentation import CANDIDATE_POLICY

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    data_root, output = Path(data_root), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous previews; choose a fresh output directory')
    train, validation = read_official_inputs(data_root)
    selection_path = data_root / 'training_inspection_selection.json'
    selection = json.loads(selection_path.read_text())
    baseline = json.loads((ROOT / 'configs/unetpp_rd_002_training.json').read_text())
    if selection['revision'] != baseline['dataset_revision'] or selection['seed'] != baseline['seed']:
        raise ValueError('Training inspection selection differs from the pinned dataset/seed')
    clips = selection['selection']
    ids = [c['clip_id'] for c in clips]
    if len(ids) != 8 or len(set(ids)) != 8 or set(ids) & set(validation['clip_id']):
        raise ValueError('Expected exactly eight unique training-only inspection clips')
    source_records = {}
    rows = []
    for clip in clips:
        row = train.loc[train['clip_id'] == clip['clip_id']]
        if len(row) != 1 or row.iloc[0]['file_path'] != clip['file_path'] or int(row.iloc[0]['label']) != clip['label']:
            raise ValueError('Preview clip identity/label differs from official training membership')
        path = data_root / 'clips' / clip['file_path']
        if path.stat().st_size != clip['size'] or file_digest(path) != clip['sha256']:
            raise ValueError(f'Preview source checksum differs: {clip["clip_id"]}')
        rows.append(row)
        source_records[clip['clip_id']] = {'file_path': clip['file_path'], 'sha256': clip['sha256'],
            'bytes': clip['size'], 'dataset_revision': selection['revision']}
    import pandas as pd
    selected = pd.concat(rows, ignore_index=True)
    output.mkdir(parents=True, exist_ok=False)
    cache = output / 'preview_inputs'
    cache.mkdir()
    paths = []
    for row in selected.itertuples(index=False):
        tensor = prepare_unetpp_clip(data_root / 'clips' / row.file_path,
            int(row.frame_count), int(row.height), int(row.width))
        path = cache / f'{row.clip_id}.npy'
        np.save(path, tensor.numpy(), allow_pickle=False)
        paths.append(path)
        del tensor
        gc.collect()
        print(f'Prepared preview input {len(paths)}/{len(selected)}: {row.clip_id}', flush=True)
    before = {p.name: file_digest(p) for p in paths}
    result = save_preview(CachedClips(paths, selected['label']), selected, ids,
        policy or CANDIDATE_POLICY, baseline['seed'], output / 'previews', source_records)
    if before != {p.name: file_digest(p) for p in paths}:
        raise ValueError('Preview modified a prepared input array')
    runtime = {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
        'opencv': cv2.__version__, 'device': 'cpu', 'source_selection_sha256': file_digest(selection_path),
        'preprocessing_source_sha256': file_digest(ROOT / 'src/rd_ultrasound_ml/unetpp_baseline.py'),
        'preview_only': True, 'local_decode_not_a_claim_of_bitwise_Kaggle_cache_identity': True,
        'input_arrays_unchanged': True, 'test_used': False}
    write_json_atomic(output / 'runtime.json', runtime)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/erdes')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/augmentation/unetpp_rd_aug_001')
    args = parser.parse_args()
    result = build(args.data, args.output)
    print(json.dumps({'clips': len(result['cases']), 'policy_digest': result['policy_digest'],
        'previews': str(args.output / 'previews'), 'test_used': False}, indent=2))
