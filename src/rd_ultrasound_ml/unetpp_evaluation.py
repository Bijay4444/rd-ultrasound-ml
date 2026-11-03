"""Frozen-checkpoint official-test evaluation; contains no training or tuning."""
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rd_ultrasound_ml.unetpp_data import CachedClips
from rd_ultrasound_ml.unetpp_kaggle import read_official_inputs
from rd_ultrasound_ml.unetpp_training import (
    autocast_for, binary_metrics, file_digest, json_digest, write_json_atomic,
)


def verify_frozen_training(directory, freeze):
    """Check the selected artifact before reading any test partition."""
    directory = Path(directory)
    if (freeze['training_epochs_completed'] != 50 or freeze['threshold'] != .5
            or freeze['test_accessed_at_freeze'] is not False or freeze['checkpoint_file'] != 'best.pt'):
        raise ValueError('Invalid pre-test freeze: require complete training and the fixed protocol')
    config = json.loads((directory/'config.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    if (json_digest(config) != freeze['training_config_digest']
            or summary['config_digest'] != freeze['training_config_digest']
            or summary['data_fingerprint'] != freeze['training_data_fingerprint']):
        raise ValueError('Frozen training configuration/data fingerprint differs')
    if (summary['status'] != 'completed' or summary['epochs_completed'] != 50
            or summary['best_epoch'] != freeze['selected_epoch'] or summary['test_used']):
        raise ValueError('Training is incomplete or selected epoch differs from freeze')
    if file_digest(directory/'best.pt') != freeze['checkpoint_sha256']:
        raise ValueError('Selected checkpoint SHA256 differs from pre-test freeze')
    return config


def find_frozen_training(input_root, freeze):
    matches = [path.parent for path in Path(input_root).rglob('best.pt')
               if file_digest(path) == freeze['checkpoint_sha256']]
    if len(matches) != 1:
        raise ValueError('Attach exactly one matching final-training output for the frozen checkpoint')
    verify_frozen_training(matches[0], freeze)
    return matches[0]


def validate_test_partition(test, metadata, train, validation):
    if (len(test) != 1077 or not test['path'].is_unique
            or test['label'].value_counts().to_dict() != {0: 977, 1: 100}):
        raise ValueError('Official test count, uniqueness or labels differ')
    if set(test['path']) & (set(train['path']) | set(validation['path'])):
        raise ValueError('Test clips overlap training or validation')
    joined = test.merge(metadata, left_on='path', right_on='file_path', how='left',
                        sort=False, validate='one_to_one')
    if (joined['clip_id'].isna().any() or not joined['clip_id'].is_unique
            or not joined['label'].map({0:'non_rd', 1:'rd'}).eq(joined['diagnostic_class']).all()
            or joined['path'].tolist() != test['path'].tolist()):
        raise ValueError('Test manifest/metadata identity, labels or order differ')
    for value in joined['file_path']:
        if Path(value).is_absolute() or '..' in Path(value).parts:
            raise ValueError('Unsafe test clip path')
    return joined


def read_frozen_test_inputs(source_directory, training_directory, freeze):
    verify_frozen_training(training_directory, freeze)  # Must precede even opening test.csv.
    source_directory = Path(source_directory)
    train, validation = read_official_inputs(source_directory)  # Checks official archive/metadata MD5s.
    with zipfile.ZipFile(source_directory/'splits.zip') as archive:
        content = archive.read('splits/non_rd_vs_rd/test.csv')
    metadata = pd.read_csv(source_directory/'erdes_metadata.csv')
    frame = validate_test_partition(pd.read_csv(io.BytesIO(content)), metadata, train, validation)
    return frame, hashlib.sha256(content).hexdigest()


def load_test_cache(input_root, training_directory, freeze, report=print):
    manifests = list(Path(input_root).rglob('test_prepared_manifest.json'))
    if len(manifests) != 1:
        raise ValueError('Attach exactly one successfully saved test-preparation output')
    root = manifests[0].parent
    frame, split_hash = read_frozen_test_inputs(root, training_directory, freeze)
    manifest = json.loads(manifests[0].read_text())
    if (manifest.get('complete') is not True or manifest['freeze_digest'] != json_digest(freeze)
            or manifest['test_csv_sha256'] != split_hash or len(manifest['entries']) != len(frame)):
        raise ValueError('Test cache is incomplete or differs from the pre-test freeze/split')
    identity = manifest['preprocessing_identity']
    if (identity['preprocessing'] != freeze['preprocessing']
            or identity['implementation'] != freeze['training_implementation_sha256']['unetpp_baseline']):
        raise ValueError('Test preprocessing differs from the frozen training method')
    paths=[]
    for index, row in enumerate(frame.itertuples(index=False)):
        entry=manifest['entries'][index]
        expected={'signature':json_digest(identity), 'clip_id':row.clip_id, 'file_path':row.file_path,
            'source_sha256':entry['source']['sha256'], 'frame_count':int(row.frame_count),
            'height':int(row.height), 'width':int(row.width), 'target':[96,128,128]}
        if (entry['index'] != index or entry['clip_id'] != row.clip_id or entry['label'] != int(row.label)
                or entry['check']['identity'] != expected):
            raise ValueError('Test cache identity/order/source provenance differs')
        relative=Path(entry['array'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe test cache path')
        path=root/relative
        if file_digest(path) != entry['check']['tensor_sha256']:
            raise ValueError(f'Test tensor checksum differs: {row.clip_id}')
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (1,96,128,128):
            raise ValueError('Test tensor shape/dtype differs')
        paths.append(path)
        if (index+1)%100==0 or index+1==len(frame):
            report(f'Verified test input {index+1}/{len(frame)}')
    return frame, CachedClips(paths,frame['label']), manifest


def evaluate_frozen_model(model_factory, dataset, frame, training_directory, freeze,
                          output_directory, runtime, device='cuda', report=print):
    """One forward pass per selected test clip, with a fixed threshold and no optimizer."""
    verify_frozen_training(training_directory,freeze)
    if len(dataset) != len(frame) or len(frame) != freeze['test_clips']:
        raise ValueError('Test dataset/manifest count differs from freeze')
    output_directory=Path(output_directory);output_directory.mkdir(parents=True,exist_ok=True)
    if (output_directory/'test_metrics.json').exists():
        raise ValueError('Test results already exist; inspect them rather than rerunning model selection')
    payload=torch.load(Path(training_directory)/'best.pt',map_location='cpu',weights_only=True)
    if (payload['epoch'] != freeze['selected_epoch'] or json_digest(payload['config']) != freeze['training_config_digest']
            or payload['data_fingerprint'] != freeze['training_data_fingerprint']):
        raise ValueError('Checkpoint payload differs from frozen selection')
    device=torch.device(device)
    model=model_factory().to(device);model.load_state_dict(payload['model_state']);model.eval()
    before={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    generator=torch.Generator().manual_seed(0)
    loader=DataLoader(dataset,batch_size=freeze['physical_batch'],shuffle=False,num_workers=0,
        pin_memory=device.type=='cuda',generator=generator)
    probabilities=[];labels=[];loss_sum=0.;start=time.perf_counter()
    criterion=torch.nn.BCEWithLogitsLoss()
    with torch.inference_mode():
        for inputs,target in loader:
            inputs,target=inputs.to(device),target.to(device)
            with autocast_for(device,freeze['precision']):
                logits=model(inputs).reshape(-1);loss=criterion(logits,target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite test inference loss')
            probabilities.extend(torch.sigmoid(logits.float()).cpu().tolist());labels.extend(target.cpu().tolist())
            loss_sum+=loss.item()*len(target)
            if len(labels)%100==0 or len(labels)==len(frame):report(f'Inferred test clips {len(labels)}/{len(frame)}')
    if not np.array_equal(np.asarray(labels),frame['label'].to_numpy()):
        raise ValueError('Inference label order differs from official test manifest')
    if any(not torch.equal(value.detach().cpu(),before[name]) for name,value in model.state_dict().items()):
        raise ValueError('Model state changed during test evaluation')
    metrics=binary_metrics(np.asarray(labels),np.asarray(probabilities),freeze['threshold'])
    prediction=frame[['clip_id','file_path','label']].copy()
    prediction['rd_probability']=probabilities
    prediction['predicted_label']=(prediction['rd_probability']>=freeze['threshold']).astype(int)
    prediction.to_csv(output_directory/'test_predictions.csv',index=False)
    frame.to_csv(output_directory/'test_manifest.csv',index=False)
    result={'evaluation_id':freeze['evaluation_id'],'partition':'official_test','test_clips':len(frame),
        'selected_epoch':freeze['selected_epoch'],'checkpoint_sha256':freeze['checkpoint_sha256'],
        'freeze_digest':json_digest(freeze),'precision':freeze['precision'],'runtime':runtime,
        'loss':loss_sum/len(frame),'seconds':time.perf_counter()-start,'metrics':metrics,
        'test_used':True,'training_performed':False,'threshold_tuned':False}
    write_json_atomic(output_directory/'test_metrics.json',result)
    write_json_atomic(output_directory/'evaluation_freeze.json',freeze)
    reference=freeze['published_reference']
    pd.DataFrame([{'source':'our frozen UNet++','partition':'official_test',**metrics},
                  {'source':'ERDES article Table 8 (author reported)','partition':'test',
                   **{key:reference[key] for key in ['sensitivity','specificity','precision','F1','accuracy']}}]).to_csv(
        output_directory/'test_reference_comparison.csv',index=False)
    return result
