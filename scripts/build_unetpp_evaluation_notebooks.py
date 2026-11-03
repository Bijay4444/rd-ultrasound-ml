"""Build separate CPU test preparation and frozen GPU inference notebooks."""
import json
from pathlib import Path
import pprint
from textwrap import dedent
ROOT=Path(__file__).resolve().parents[1]

def build():
    freeze=json.loads((ROOT/'configs/unetpp_rd_002_evaluation.json').read_text())
    training=json.loads((ROOT/'notebooks/kaggle_unetpp_training.ipynb').read_text())
    shared={c['id']:''.join(c['source']) for c in training['cells']}
    old=json.loads((ROOT/'notebooks/unetpp_training.ipynb').read_text())
    download=''.join(next(c for c in old['cells'] if c['id']=='training-download-helper')['source'])
    for mode,filename in [('prepare','kaggle_unetpp_test_preparation.ipynb'),('evaluate','kaggle_unetpp_evaluation.ipynb')]:
        cells=[]
        def add(kind,ident,source):
            c={'cell_type':kind,'id':ident,'metadata':{},'source':dedent(source).lstrip('\n').splitlines(keepends=True)}
            if c['source']:c['source'][-1]=c['source'][-1].rstrip('\n')+'\n'
            if kind=='code':c.update(execution_count=None,outputs=[])
            cells.append(c)
        add('markdown','evaluation-intro','''
# UNet++ RD: official-test input preparation

Training completed all 50 epochs. Epoch 48 was selected by highest validation
accuracy, with earliest ties; its checkpoint SHA256 and threshold 0.5 were frozen
in `configs/unetpp_rd_002_evaluation.json` **before the test CSV was opened**.
Run this as a private saved **CPU** job with Internet enabled, attaching final
training job 03 version 1. It verifies the frozen checkpoint, then prepares all
1,077 official test clips using the unchanged float32 `[1,96,128,128]` method.
The saved cache is about 6.78 GB. It performs no model inference or training.

Do not alter labels, threshold, selected epoch or preprocessing based on these
inputs. Only after this CPU output saves successfully, attach it and final
training job 03 to the separate GPU evaluation notebook. Keep the freeze file.
''' if mode=='prepare' else '''
# UNet++ RD: frozen official-test evaluation

This is a separate final-test stage after all 50 training epochs. It attaches
one saved CPU test-input output and the frozen final-training output, verifies
the checkpoint SHA256, then evaluates **epoch 48 at threshold 0.5** on all 1,077
official test clips. It contains no optimizer, retraining, threshold search,
calibration fit or model-selection loop. No videos are downloaded here.

Run once as a private saved GPU job. Preserve the resulting predictions, metrics,
freeze and ZIP before interpreting the comparison. Numerical AMP, decoder and
single-GPU adaptations are retained and disclosed. The article's reported test
row is a reference, not a target for tuning. Dataset splits are clip-level;
these educational results do not establish clinical or unseen-patient validity.
''')
        add('code','evaluation-runtime',shared['kaggle-runtime'])
        add('code','evaluation-freeze','UNETPP_FREEZE = '+pprint.pformat(freeze,sort_dicts=False,width=100)+'\n')
        add('code','evaluation-module-loader',shared['kaggle-module-loader'])
        for name in ['unetpp_baseline','unetpp_training','unetpp_data','unetpp_kaggle','unetpp_evaluation']:
            source=(ROOT/f'src/rd_ultrasound_ml/{name}.py').read_text()
            add('code','evaluation-embed-'+name,'UNETPP_MODULE_SOURCE = '+repr(source)+'\nunetpp_load_module('+repr(name)+', UNETPP_MODULE_SOURCE)\n')
        add('code','evaluation-imports','''
from rd_ultrasound_ml.unetpp_baseline import UNetPPRDClassifier
from rd_ultrasound_ml.unetpp_training import file_digest, json_digest, write_json_atomic
from rd_ultrasound_ml.unetpp_data import prepare_cached_clip, record_video_sources
from rd_ultrasound_ml.unetpp_evaluation import (find_frozen_training, verify_frozen_training,
    read_frozen_test_inputs, load_test_cache, evaluate_frozen_model)
for name,digest in UNETPP_FREEZE['training_implementation_sha256'].items():
    if hashlib.sha256(unetpp_sources_embedded[name].encode()).hexdigest() != digest:
        raise ValueError('Embedded training implementation differs from freeze: '+name)
unetpp_training_dir=find_frozen_training(Path('/kaggle/input'),UNETPP_FREEZE)
unetpp_training_config=verify_frozen_training(unetpp_training_dir,UNETPP_FREEZE)
print('Frozen training verified: epoch',UNETPP_FREEZE['selected_epoch'],'| checkpoint SHA256:',UNETPP_FREEZE['checkpoint_sha256'])
''')
        if mode=='prepare':
            add('code','evaluation-cpu-storage','''
if torch.cuda.is_available():
    raise RuntimeError('Test input preparation must use Accelerator None to preserve GPU quota.')
unetpp_output=Path('/kaggle/working/unetpp_test_inputs')
unetpp_output.mkdir(parents=True,exist_ok=True)
write_json_atomic(unetpp_output/'evaluation_freeze.json',UNETPP_FREEZE)
write_json_atomic(unetpp_output/'runtime.json',unetpp_runtime)
unetpp_identity={'preprocessing':UNETPP_FREEZE['preprocessing'],
    'implementation':UNETPP_FREEZE['training_implementation_sha256']['unetpp_baseline'],
    'torch':str(torch.__version__),'opencv':cv2.__version__,'numpy':np.__version__}
unetpp_original_identity=json.loads((unetpp_training_dir/'input_fingerprint.json').read_text())['preprocessing_identity']
if unetpp_identity != unetpp_original_identity:
    raise ValueError('CPU preprocessing versions/method differ from the saved training inputs; inspect before test access.')
''')
            add('code','evaluation-download-helper',download)
            add('code','evaluation-test-partition','''
unetpp_release='https://zenodo.org/records/18644370'
download_checked(f'{unetpp_release}/files/erdes_metadata.csv?download=1',unetpp_output/'erdes_metadata.csv',
    '95828fb5bdffdbf8173b5bcfc99e8148',algorithm='md5')
download_checked(f'{unetpp_release}/files/splits.zip?download=1',unetpp_output/'splits.zip',
    'c5630e3f056ce358dcd384aaed946e64',algorithm='md5')
unetpp_test,unetpp_test_csv_hash=read_frozen_test_inputs(unetpp_output,unetpp_training_dir,UNETPP_FREEZE)
unetpp_test.to_csv(unetpp_output/'test_manifest.csv',index=False)
unetpp_sources=record_video_sources(unetpp_test,UNETPP_FREEZE['dataset_revision'],
    unetpp_output/'video_sources.json',HfApi(token=False),report=lambda value:print(value,flush=True))
unetpp_cache=unetpp_output/'arrays'
unetpp_needed=len(unetpp_test)*(96*128*128*4+128)+500_000_000
unetpp_existing=sum(path.stat().st_size for path in unetpp_cache.glob('*.npy')) if unetpp_cache.exists() else 0
if unetpp_needed>19_000_000_000 or shutil.disk_usage(unetpp_output).free<max(0,unetpp_needed-unetpp_existing):
    raise RuntimeError('Insufficient saved-output space for test preparation.')
print('Official test clips:',len(unetpp_test),'| labels:',unetpp_test['label'].value_counts().to_dict())
''')
            add('code','evaluation-prepare-test','''
unetpp_entries=[]
with TemporaryDirectory(prefix='unetpp_test_raw_',dir='/tmp') as unetpp_raw:
    for index,row in enumerate(unetpp_test.itertuples(index=False)):
        source=unetpp_sources[row.file_path]
        path,check=prepare_cached_clip(row,source,unetpp_raw,unetpp_cache,json_digest(unetpp_identity),download_checked)
        raw=Path(unetpp_raw)/'clips'/row.file_path
        if raw.exists():raw.unlink()  # Owned temporary download, excluded from saved output.
        unetpp_entries.append({'index':index,'clip_id':row.clip_id,'label':int(row.label),
            'array':str(path.relative_to(unetpp_output)),'source':source,'check':check})
        if index==0 or (index+1)%10==0 or index+1==len(unetpp_test):
            print(f'Prepared official test clip {index+1}/{len(unetpp_test)}',flush=True)
write_json_atomic(unetpp_output/'test_prepared_manifest.json',{'schema_version':1,'complete':True,
    'freeze_digest':json_digest(UNETPP_FREEZE),'test_csv_sha256':unetpp_test_csv_hash,
    'preprocessing_identity':unetpp_identity,'entries':unetpp_entries})
write_json_atomic(unetpp_output/'test_preparation_summary.json',{'complete':True,'test_clips':len(unetpp_test),
    'dtype':'float32','gpu_used':False,'model_inference_performed':False,'freeze_digest':json_digest(UNETPP_FREEZE)})
print('TEST PREPARATION COMPLETE. Wait for saved output, then submit the separate GPU evaluation job.')
''')
        else:
            add('code','evaluation-verified-inputs','''
if not torch.cuda.is_available():raise RuntimeError('Select a Kaggle CUDA GPU for frozen test inference.')
if json.loads((unetpp_training_dir/'runtime.json').read_text()) != unetpp_runtime:
    raise ValueError('GPU runtime versions/settings differ from frozen training; inspect before inference.')
unetpp_test,unetpp_test_dataset,unetpp_manifest=load_test_cache(
    Path('/kaggle/input'),unetpp_training_dir,UNETPP_FREEZE,report=lambda value:print(value,flush=True))
unetpp_output=Path('/kaggle/working/unetpp_rd_002_test')
unetpp_output.mkdir(parents=True,exist_ok=True)
write_json_atomic(unetpp_output/'runtime.json',unetpp_runtime)
write_json_atomic(unetpp_output/'test_input_provenance.json',{'freeze_digest':json_digest(UNETPP_FREEZE),
    'test_csv_sha256':unetpp_manifest['test_csv_sha256'],
    'prepared_manifest_sha256':json_digest(unetpp_manifest),
    'preprocessing_identity':unetpp_manifest['preprocessing_identity']})
for name,source in unetpp_sources_embedded.items():(unetpp_output/(name+'.py')).write_text(source)
print('READY for frozen inference:',len(unetpp_test_dataset),'official test clips.')
''')
            add('code','evaluation-inference','''
unetpp_test_result=evaluate_frozen_model(UNetPPRDClassifier,unetpp_test_dataset,unetpp_test,
    unetpp_training_dir,UNETPP_FREEZE,unetpp_output,unetpp_runtime,
    device='cuda',report=lambda value:print(value,flush=True))
print('OFFICIAL TEST METRICS:',unetpp_test_result['metrics'])
display(pd.read_csv(unetpp_output/'test_reference_comparison.csv'))
print('Reference values are author-reported. This single adapted run does not establish identical experimental conditions or clinical validity.')
''')
            add('code','evaluation-export','''
unetpp_hashes={path.name:file_digest(path) for path in sorted(unetpp_output.iterdir())
    if path.is_file() and path.suffix in {'.json','.csv','.py'} and path.name!='artifact_sha256.json'}
write_json_atomic(unetpp_output/'artifact_sha256.json',unetpp_hashes)
unetpp_zip=unetpp_output.parent/'unetpp_rd_002_test_results.zip'
with zipfile.ZipFile(unetpp_zip,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in [*unetpp_hashes,'artifact_sha256.json']:archive.write(unetpp_output/name,arcname=f'{unetpp_output.name}/{name}')
print('Saved test results ZIP:',unetpp_zip)
print('Wait for this saved job to complete, then preserve its Output ZIP. No model was trained or tuned in this notebook.')
''')
        nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python'},'kaggle':{'isGpuEnabled':mode=='evaluate','isInternetEnabled':mode=='prepare'}},
            'nbformat':4,'nbformat_minor':5}
        (ROOT/'notebooks'/filename).write_text(json.dumps(nb,indent=1)+'\n')

if __name__=='__main__':build()
