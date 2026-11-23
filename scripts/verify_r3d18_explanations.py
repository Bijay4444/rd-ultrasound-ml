"""Audit the saved R3D explanation extension without new model inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from build_explainability_demo import verified_archive, file_sha

ROOT=Path(__file__).resolve().parents[1]


def verify(archive_path, reference_archive, training, output):
    output=Path(output)
    plan=json.loads((ROOT/'configs/r3d18_gradcam_validation_001.json').read_text())
    freeze=json.loads((ROOT/'configs/r3d18_rd_001_evaluation.json').read_text())
    predictions=json.loads((Path(training)/'best_validation_predictions.json').read_text())
    with zipfile.ZipFile(reference_archive) as old, zipfile.ZipFile(archive_path) as new:
        old_prefix,old_hashes=verified_archive(old)
        prefix,hashes=verified_archive(new)
        def read(name):return json.loads(new.read(prefix+name))
        def require(condition, message):
            if not condition:raise ValueError(message)
        require(file_sha(reference_archive)==plan['reference_archive_sha256'],'Historical archive identity')
        status,selection=read('status.json'),read('selection_manifest.json')
        require(status['status']=='completed' and set(status['models'])=={'r3d18'}
                and status['models']['r3d18']['status']=='completed','Completed R3D-only status')
        require(selection==plan['selection']==json.loads(old.read(old_prefix+'selection_manifest.json')),
                'Original fixed eight-case panel')
        require(read('settings.json')==json.loads(old.read(old_prefix+'settings.json')),'Unchanged method/cache settings')
        require(new.read(prefix+'sources/video_explainability.py')==old.read(old_prefix+'sources/video_explainability.py'),
                'Unchanged Grad-CAM/control algorithm')
        require(read('r3d18/freeze.json')==freeze,'Exact selected-model freeze')
        config=read('r3d18/training_config.json')
        require(hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                ==freeze['training_config_digest'],'Resolved training configuration')
        for name in hashes:
            if name.startswith('sources/') and name.endswith('.py'):
                require(new.read(prefix+name)==(ROOT/'src/rd_ultrasound_ml'/Path(name).name).read_bytes(),
                        'Current embedded source: '+name)
        for module,digest in freeze['training_implementation_sha256'].items():
            require(hashes['sources/'+module+'.py']==digest,'Frozen training source: '+module)
        verified=read('verified_models.json')
        require(verified=={'verified':['unetpp','r3d18'],'explained':['r3d18'],
                          'fixed_historical_cases_checked':True},'Verified reference and explained subset')
        rows,parity=read('case_results.json'),read('r3d18/prediction_parity.json')
        expected=[c['clip_id'] for c in selection['cases']]
        require(len(rows)==len(parity)==len(status['cases'])==8,'Eight complete model/case records')
        require([r['clip_id'] for r in rows]==[r['clip_id'] for r in parity]==expected,'Ordered case coverage')
        require([r['clip_id'] for r in status['cases']]==expected
                and all(r['model']=='r3d18' and r['status']=='completed' for r in status['cases']),
                'Status case coverage')
        maps_checked=0
        controls=[]
        for case,row,p in zip(selection['cases'],rows,parity):
            key='r3d18/'+case['clip_id']+'/'
            require(all(row[k]==v and p[k]==v for k,v in case.items()),'Case identity')
            require(row['model']=='r3d18' and read(key+'result.json')==row,'Per-case result identity')
            index=case['validation_index']
            require(p['saved_probability']==predictions['rd_probabilities'][index]
                    and predictions['dataset_labels'][index]==case['label'],'Original saved validation probability')
            require(abs(p['original_precision_probability']-p['saved_probability'])<=.001,'Historical probability parity')
            require(row['fp32_probability']==p['fp32_probability'],'FP32 score record')
            outcome=('TP' if p['fp32_probability']>=.5 else 'FN') if case['label'] else ('FP' if p['fp32_probability']>=.5 else 'TN')
            require(row['outcome']==outcome,'Fixed-threshold case outcome')
            for target,sign,record in zip(('rd','non_rd'),(1,-1),row['maps']):
                with np.load(io.BytesIO(new.read(prefix+key+target+'.npz')),allow_pickle=False) as arrays:
                    raw,normalized=arrays['raw'],arrays['normalized']
                    require(raw.shape==normalized.shape==(96,128,128),'Full-resolution attribution shape')
                    require(np.isfinite(raw).all() and np.isfinite(normalized).all()
                            and raw.min()>=0 and normalized.min()>=0 and normalized.max()<=1,'Finite attribution range')
                    span=raw.max()-raw.min()
                    constant=bool(span<=1e-12)
                    expected_map=np.zeros_like(raw) if constant else (raw-raw.min())/span
                    require(np.allclose(normalized,expected_map,rtol=0,atol=1e-6),'Whole-clip normalization')
                    require(record['constant_map']==constant and record['zero_map']==bool(raw.max()<=1e-12)
                            and record['target_sign']==sign and record['activation_shape']==[1,512,12,8,8],
                            'Target, constant/zero flags and feature shape')
                    if target=='rd':rd_reference=normalized.copy()
                    maps_checked+=1
            occlusion,randomization=read(key+'occlusion.json'),read(key+'randomization.json')
            require(occlusion['cuboid']==[12,16,16] and occlusion['count']==8 and occlusion['seed']==42,
                    'Fixed occlusion configuration')
            for group,record in occlusion['groups'].items():
                require(len(record['interventions'])==8,'Eight occlusions per group')
                drops=[occlusion['baseline_logit']-r['logit'] for r in record['interventions']]
                require(np.allclose(drops,[r['drop'] for r in record['interventions']],rtol=0,atol=1e-12)
                        and np.isclose(np.mean(drops),record['mean_drop'],rtol=0,atol=1e-12),'Occlusion arithmetic')
            require([r['stage'] for r in randomization]==['head','head_and_block','all'],'Cumulative randomization stages')
            for r in randomization:
                with np.load(io.BytesIO(new.read(prefix+key+r['stage']+'.npz')),allow_pickle=False) as arrays:
                    require(arrays['raw'].shape==arrays['normalized'].shape==(96,128,128)
                            and np.isfinite(arrays['raw']).all() and np.isfinite(arrays['normalized']).all(),
                            'Finite randomized attribution arrays')
                    a,b=rd_reference.ravel(),arrays['normalized'].ravel()
                    correlation=None if np.std(a)<1e-12 or np.std(b)<1e-12 else float(np.corrcoef(a,b)[0,1])
                    saved=r['similarity']
                    require((saved['pearson'] is None if correlation is None else saved['pearson'] is not None
                            and np.isclose(correlation,saved['pearson'],rtol=0,atol=1e-12))
                            and np.isclose(float(np.abs(a-b).mean()),saved['mean_absolute_difference'],rtol=0,atol=1e-12),
                            'Recomputed map similarity')
                maps_checked+=1
            controls.append({'clip_id':case['clip_id'],'constant_rd_map':occlusion['constant_map'],
                             **{k:v['mean_drop'] for k,v in occlusion['groups'].items()}})
        output.mkdir(parents=True,exist_ok=False)
        for name in [*hashes,'artifact_sha256.json']:
            destination=output/'verified'/prefix/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(new.read(prefix+name))
        report={'status':'passed','verified_at_utc':datetime.now(timezone.utc).isoformat(),
            'archive_sha256':file_sha(archive_path),'artifact_hashes_verified':len(hashes),
            'historical_artifact_hashes_verified':len(old_hashes),'cases':8,'attribution_arrays_checked':maps_checked,
            'selected_epoch':18,'checkpoint_sha256':freeze['checkpoint_sha256'],
            'max_historical_probability_difference':max(p['original_vs_saved'] for p in parity),
            'max_fp32_probability_difference':max(p['fp32_vs_original'] for p in parity),
            'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rows),
            'runtime':read('runtime.json'),'seconds':status['seconds'],'controls':controls,
            'model_inference_performed_by_audit':False,'official_test_data_used':False}
        (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--reference-archive',type=Path,default=ROOT/'outputs/kaggle_results/gradcam_validation_001/gradcam_validation_001.zip')
    parser.add_argument('--training',type=Path,default=ROOT/'outputs/kaggle_peer_audit/r3d18/training/r3d18_rd_001_amp')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    verify(args.archive,args.reference_archive,args.training,args.output)
