"""Frozen-test boundaries and actual inference behavior on synthetic CPU inputs."""
import ast
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from rd_ultrasound_ml.unetpp_evaluation import (
    verify_frozen_training, find_frozen_training, validate_test_partition,
    read_frozen_test_inputs, load_test_cache, evaluate_frozen_model,
)
from rd_ultrasound_ml.unetpp_training import json_digest,file_digest,write_json_atomic
from prepare_unetpp_evaluation_job import write_evaluation_bundle


class UNetPPEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def training_fixture(self,root):
        directory=Path(root)/'training';directory.mkdir()
        config={'seed':42,'evaluation':{'threshold':.5}}
        model=torch.nn.Linear(1,1)
        with torch.no_grad():model.weight.fill_(5);model.bias.zero_()
        torch.save({'epoch':48,'config':config,'data_fingerprint':'fixed','model_state':model.state_dict()},directory/'best.pt')
        write_json_atomic(directory/'config.json',config)
        write_json_atomic(directory/'summary.json',{'status':'completed','epochs_completed':50,'best_epoch':48,
            'config_digest':json_digest(config),'data_fingerprint':'fixed','test_used':False})
        freeze={'training_epochs_completed':50,'threshold':.5,'test_accessed_at_freeze':False,'checkpoint_file':'best.pt',
            'training_config_digest':json_digest(config),'training_data_fingerprint':'fixed','selected_epoch':48,
            'checkpoint_sha256':file_digest(directory/'best.pt'),'test_clips':4,'evaluation_id':'synthetic_only',
            'physical_batch':2,'precision':'float32','preprocessing':{'fixture':True},
            'training_implementation_sha256':{'unetpp_baseline':'a'*64},
            'published_reference':{'sensitivity':.879,'specificity':.986,'precision':.871,'F1':.875,'accuracy':.976}}
        return directory,freeze

    def test_freeze_rejects_changed_checkpoint_or_selection_before_test_access(self):
        with TemporaryDirectory() as tmp:
            directory,freeze=self.training_fixture(tmp)
            verify_frozen_training(directory,freeze)
            self.assertEqual(find_frozen_training(tmp,freeze),directory)
            altered={**freeze,'selected_epoch':50}
            with self.assertRaisesRegex(ValueError,'selected epoch'):
                verify_frozen_training(directory,altered)
            (directory/'best.pt').write_bytes(b'changed model')
            with patch('rd_ultrasound_ml.unetpp_evaluation.read_official_inputs') as read_split:
                with self.assertRaisesRegex(ValueError,'SHA256'):
                    read_frozen_test_inputs('/must/not/read',directory,freeze)
                read_split.assert_not_called()

    def test_official_test_membership_labels_order_and_disjointness(self):
        test=pd.DataFrame({'path':[f'test/{i}.mp4' for i in range(1077)],'label':[0]*977+[1]*100})
        metadata=pd.DataFrame({'file_path':test['path'],'clip_id':[f'clip{i}' for i in range(1077)],
            'diagnostic_class':test['label'].map({0:'non_rd',1:'rd'})})
        train=pd.DataFrame({'path':['train/a.mp4']});val=pd.DataFrame({'path':['val/b.mp4']})
        result=validate_test_partition(test,metadata,train,val)
        self.assertEqual(result['file_path'].tolist(),test['path'].tolist())
        with self.assertRaisesRegex(ValueError,'overlap'):
            validate_test_partition(test,metadata,test.iloc[:1],val)
        metadata.loc[0,'diagnostic_class']='rd'
        with self.assertRaisesRegex(ValueError,'labels'):
            validate_test_partition(test,metadata,train,val)

    def test_actual_inference_uses_frozen_weights_and_threshold_without_updates(self):
        with TemporaryDirectory() as tmp:
            directory,freeze=self.training_fixture(tmp)
            frame=pd.DataFrame({'clip_id':['a','b','c','d'],'file_path':['a','b','c','d'],'label':[0,1,0,1]})
            dataset=TensorDataset(torch.tensor([[-1.],[1.],[-.5],[.5]]),torch.tensor([0.,1.,0.,1.]))
            output=Path(tmp)/'evaluation'
            before=file_digest(directory/'best.pt')
            with patch('torch.optim.Adam',side_effect=AssertionError('No optimizer may be constructed')):
                result=evaluate_frozen_model(lambda:torch.nn.Linear(1,1),dataset,frame,directory,freeze,
                    output,{},device='cpu',report=lambda _:None)
            self.assertEqual((result['metrics']['TP'],result['metrics']['TN'],result['metrics']['FP'],result['metrics']['FN']),(2,2,0,0))
            self.assertEqual(file_digest(directory/'best.pt'),before)
            self.assertFalse(result['training_performed']);self.assertFalse(result['threshold_tuned'])
            prediction=pd.read_csv(output/'test_predictions.csv')
            self.assertEqual(prediction['clip_id'].tolist(),frame['clip_id'].tolist())
            self.assertEqual(prediction['predicted_label'].tolist(),frame['label'].tolist())
            with self.assertRaisesRegex(ValueError,'already exist'):
                evaluate_frozen_model(lambda:torch.nn.Linear(1,1),dataset,frame,directory,freeze,output,{},device='cpu')

    def test_test_cache_rejects_tensor_corruption_and_reordered_manifest(self):
        with TemporaryDirectory() as tmp:
            directory,freeze=self.training_fixture(tmp)
            cache=Path(tmp)/'input/cache';cache.mkdir(parents=True)
            frame=pd.DataFrame({'clip_id':['a','b','c','d'],'file_path':['a','b','c','d'],'label':[0,1,0,1],
                'frame_count':[3]*4,'height':[32]*4,'width':[48]*4})
            identity={'preprocessing':freeze['preprocessing'],'implementation':'a'*64}
            entries=[]
            for index,row in enumerate(frame.itertuples(index=False)):
                path=cache/(row.clip_id+'.npy');np.save(path,np.zeros((1,96,128,128),np.float32))
                entries.append({'index':index,'clip_id':row.clip_id,'label':row.label,'array':path.name,
                    'source':{'sha256':'b'*64},'check':{'tensor_sha256':file_digest(path),'identity':{
                    'signature':json_digest(identity),'clip_id':row.clip_id,'file_path':row.file_path,'source_sha256':'b'*64,
                    'frame_count':3,'height':32,'width':48,'target':[96,128,128]}}})
            manifest={'complete':True,'freeze_digest':json_digest(freeze),'test_csv_sha256':'fixed',
                'preprocessing_identity':identity,'entries':entries}
            write_json_atomic(cache/'test_prepared_manifest.json',manifest)
            with patch('rd_ultrasound_ml.unetpp_evaluation.read_frozen_test_inputs',return_value=(frame,'fixed')):
                _,dataset,_=load_test_cache(Path(tmp)/'input',directory,freeze,report=lambda _:None)
                self.assertEqual(len(dataset),4)
                (cache/'c.npy').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'checksum'):
                    load_test_cache(Path(tmp)/'input',directory,freeze,report=lambda _:None)

    def test_notebooks_and_version_pinned_private_bundles_match_freeze(self):
        for filename in ['kaggle_unetpp_test_preparation.ipynb','kaggle_unetpp_evaluation.ipynb']:
            notebook=json.loads((ROOT/'notebooks'/filename).read_text())
            cells={c['id']:''.join(c['source']) for c in notebook['cells']}
            freeze=ast.literal_eval(ast.parse(cells['evaluation-freeze']).body[0].value)
            self.assertEqual(freeze,json.loads((ROOT/'configs/unetpp_rd_002_evaluation.json').read_text()))
            for name in ['unetpp_baseline','unetpp_training','unetpp_data','unetpp_kaggle','unetpp_evaluation']:
                source=ast.literal_eval(ast.parse(cells['evaluation-embed-'+name]).body[0].value)
                self.assertEqual(source,(ROOT/f'src/rd_ultrasound_ml/{name}.py').read_text())
            for cell in notebook['cells']:
                if cell['cell_type']=='code':compile(''.join(cell['source']),filename,'exec')
        owner=freeze['training_source'].split('/')[0]
        with TemporaryDirectory() as tmp:
            _,cpu=write_evaluation_bundle('prepare',owner,tmp)
            _,gpu=write_evaluation_bundle('evaluate',owner,tmp)
            self.assertTrue(cpu['is_private']);self.assertFalse(cpu['enable_gpu']);self.assertTrue(cpu['enable_internet'])
            self.assertTrue(gpu['is_private']);self.assertTrue(gpu['enable_gpu']);self.assertFalse(gpu['enable_internet'])
            self.assertTrue(all(source.endswith('/1') for source in gpu['kernel_sources']))
            runtime_pin=json.loads((ROOT/'configs/unetpp_rd_002_runtime.json').read_text())
            self.assertEqual(gpu['docker_image'],runtime_pin['docker_image'])
            self.assertEqual(gpu['docker_image_pinning_type'],'original')
            self.assertNotIn('docker_image',cpu)


if __name__=='__main__':unittest.main()
