"""Kaggle handoff tests without a Kaggle account, private data or CUDA."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from rd_ultrasound_ml.unetpp_kaggle import shard_indices, load_prepared_shards, import_previous_run
from rd_ultrasound_ml.unetpp_training import CheckpointStore, file_digest, json_digest, write_json_atomic
from prepare_kaggle_job import write_bundle
from kaggle_cli import credential_environment
import kaggle_cli


class UNetPPKaggleTests(unittest.TestCase):
    def test_gpu_submission_is_blocked_when_preparation_job_has_failed(self):
        with TemporaryDirectory() as tmp:
            folder, _ = write_bundle('exampleuser', 'train', 1, tmp)
            failed = SimpleNamespace(returncode=0, stdout='exampleuser/unetpp-rd-002-inputs-00 has status "KernelWorkerStatus.ERROR"', stderr='')
            with patch.object(sys, 'argv', ['kaggle_cli.py', 'kernels', 'push', '-p', str(folder)]), \
                    patch.object(kaggle_cli.subprocess, 'run', return_value=failed), \
                    patch.object(kaggle_cli.subprocess, 'call', return_value=0) as submit, \
                    patch.object(kaggle_cli, 'credential_environment', return_value={}):
                self.assertEqual(kaggle_cli.main(), 1)
                submit.assert_not_called()

    def test_gpu_submission_checks_every_dependency_before_starting(self):
        with TemporaryDirectory() as tmp:
            folder, metadata = write_bundle('exampleuser', 'train', 2, tmp)
            seen = []
            def status(command, **kwargs):
                seen.append(command[-1])
                return SimpleNamespace(returncode=0, stdout=f'{command[-1]} has status "KernelWorkerStatus.COMPLETE"', stderr='')
            with patch.object(sys, 'argv', ['kaggle_cli.py', 'kernels', 'push', '--path='+str(folder)]), \
                    patch.object(kaggle_cli.subprocess, 'run', side_effect=status), \
                    patch.object(kaggle_cli.subprocess, 'call', return_value=0) as submit, \
                    patch.object(kaggle_cli, 'credential_environment', return_value={}):
                self.assertEqual(kaggle_cli.main(), 0)
                self.assertEqual(seen, metadata['kernel_sources'])
                submit.assert_called_once()

    def test_two_shards_preserve_every_selected_index_and_fit_saved_output_budget(self):
        first, second = shard_indices(4304, 0), shard_indices(4304, 1)
        self.assertEqual(first+second, list(range(4304)))
        self.assertFalse(set(first)&set(second))
        for indices in [first, second]:
            self.assertLess(len(indices)*(96*128*128*4+128)+500_000_000, 20_000_000_000)
        with self.assertRaises(ValueError): shard_indices(4304, 2)

    def test_private_cpu_and_gpu_submission_dependencies_and_no_implicit_submission(self):
        with TemporaryDirectory() as tmp:
            cpu, meta = write_bundle('ExampleUser', 'prepare', 1, tmp)
            self.assertFalse(meta['enable_gpu'])
            self.assertTrue(meta['enable_internet'])
            self.assertTrue(meta['is_private'])
            gpu, train = write_bundle('ExampleUser', 'train', 2, tmp, epochs=5)
            self.assertTrue(train['enable_gpu'])
            self.assertFalse(train['enable_internet'])
            self.assertEqual(train['machine_shape'], 'NvidiaTeslaT4')
            self.assertEqual(train['kernel_sources'], ['exampleuser/unetpp-rd-002-inputs-00',
                'exampleuser/unetpp-rd-002-inputs-01', 'exampleuser/unetpp-rd-002-train-01'])
            nb = json.loads((gpu/'kaggle_unetpp_training.ipynb').read_text())
            settings = ''.join(next(c for c in nb['cells'] if c['id']=='kaggle-settings')['source'])
            values = {}
            exec(settings, values)
            self.assertEqual(values['UNETPP_EPOCHS_THIS_SESSION'], 5)
            self.assertTrue(values['UNETPP_REQUIRE_PREVIOUS'])
            self.assertEqual(train['title'].lower().replace(' ', '-'), train['id'].split('/')[1])
            with self.assertRaisesRegex(ValueError, 'differs'):
                write_bundle('ExampleUser', 'train', 2, tmp, epochs=4)

    def test_credentials_are_parsed_as_data_and_never_executed(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'.env.kaggle'
            token = 'literal-$(echo secret)-`touch never-created`'
            path.write_text("KAGGLE_USERNAME='exampleuser'\nKAGGLE_API_TOKEN='"+token+"'\n")
            environment = credential_environment(path, {})
            self.assertEqual(environment['KAGGLE_API_TOKEN'], token)
            self.assertEqual(credential_environment(path, {'KAGGLE_API_TOKEN': 'existing'})['KAGGLE_API_TOKEN'], 'existing')
            path.write_text('HOME=/tmp/unsafe\n')
            with self.assertRaises(ValueError): credential_environment(path, {})

    def fixture_shards(self, directory):
        frame = pd.DataFrame({'clip_id': ['a','b','c','d'], 'file_path': ['a.mp4','b.mp4','c.mp4','d.mp4'],
            'path': ['a.mp4','b.mp4','c.mp4','d.mp4'], 'label': [0,1,0,1],
            'frame_count': [3]*4, 'height': [32]*4, 'width': [48]*4})
        train, validation = frame.iloc[:2].reset_index(drop=True), frame.iloc[2:].reset_index(drop=True)
        config, identity = {'seed':42}, {'preprocessing':'fixture_float32'}
        roots = []
        for index in range(2):
            root = Path(directory)/str(index)
            root.mkdir()
            roots.append(root)
            entries=[]
            for position in shard_indices(4,index):
                row=frame.iloc[position]
                array_path=root/(row['clip_id']+'.npy')
                np.save(array_path,np.full((1,96,128,128),position/4,dtype=np.float32))
                source={'sha256':'a'*64}
                check={'identity':{'signature':json_digest(identity),'clip_id':row['clip_id'],
                    'file_path':row['file_path'],'source_sha256':source['sha256'],'frame_count':3,
                    'height':32,'width':48,'target':[96,128,128]},'tensor_sha256':file_digest(array_path)}
                entries.append({'global_index':position,'clip_id':row['clip_id'],'label':int(row['label']),
                    'source':source,'check':check,'array':array_path.name})
            write_json_atomic(root/'prepared_manifest.json', {'complete':True,'shard_count':2,
                'shard_index':index,'config_digest':json_digest(config),'preprocessing_identity':identity,'entries':entries})
        return config, train, validation, roots

    def test_attached_shards_keep_order_fingerprint_and_reject_corruption(self):
        with TemporaryDirectory() as tmp:
            config, train, val, roots = self.fixture_shards(tmp)
            with patch('rd_ultrasound_ml.unetpp_kaggle.read_official_inputs',return_value=(train,val)):
                first=load_prepared_shards(tmp,config,report=lambda _:None)
                self.assertEqual([path.stem for path in first[2]],['a','b','c','d'])
                again=load_prepared_shards(tmp,config,report=lambda _:None)
                self.assertEqual(first[3],again[3])
                (roots[1]/'c.npy').write_bytes(b'broken')
                with self.assertRaisesRegex(ValueError,'checksum'):
                    load_prepared_shards(tmp,config,report=lambda _:None)

    def test_missing_duplicate_or_reordered_shards_stop_before_training(self):
        with TemporaryDirectory() as tmp:
            config,train,val,roots=self.fixture_shards(tmp)
            with patch('rd_ultrasound_ml.unetpp_kaggle.read_official_inputs',return_value=(train,val)):
                path=roots[1]/'prepared_manifest.json'
                manifest=json.loads(path.read_text())
                manifest['entries'].reverse()
                write_json_atomic(path,manifest)
                with self.assertRaisesRegex(ValueError,'reordered'):
                    load_prepared_shards(tmp,config,report=lambda _:None)
                path.unlink()
                with self.assertRaisesRegex(ValueError,'exactly two'):
                    load_prepared_shards(tmp,config,report=lambda _:None)

    def test_previous_saved_output_is_required_and_copied_without_modification(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'input/prior/unetpp_rd_002_amp'
            source.mkdir(parents=True)
            config={'seed':42}
            write_json_atomic(source/'config.json',config)
            write_json_atomic(source/'validation_epoch_001.json',{'probabilities':[.2,.7]})
            CheckpointStore(source).save({'model':torch.tensor([1.,2.]),'epoch':1})
            before={p.name:file_digest(p) for p in source.iterdir() if p.is_file()}
            imported=import_previous_run(root/'input',root/'working',config,True)
            self.assertEqual(imported,source)
            torch.testing.assert_close(CheckpointStore(root/'working').load()['model'],torch.tensor([1.,2.]))
            self.assertEqual(before,{p.name:file_digest(p) for p in source.iterdir() if p.is_file()})
            self.assertTrue((root/'working/validation_epoch_001.json').exists())
            with self.assertRaisesRegex(ValueError,'requires'):
                import_previous_run(root/'empty',root/'new',config,True)

    def test_notebooks_embed_current_sources_and_fixed_training_config(self):
        for name in ['kaggle_unetpp_preparation','kaggle_unetpp_training']:
            notebook=json.loads((ROOT/f'notebooks/{name}.ipynb').read_text())
            cells={c['id']:''.join(c['source']) for c in notebook['cells']}
            config=ast.literal_eval(ast.parse(cells['kaggle-config']).body[0].value)
            self.assertEqual(config,json.loads((ROOT/'configs/unetpp_rd_002_training.json').read_text()))
            self.assertEqual(config['training']['epochs'],50)
            for module in ['unetpp_baseline','unetpp_training','unetpp_data','unetpp_kaggle']:
                embedded=ast.literal_eval(ast.parse(cells['kaggle-embed-'+module]).body[0].value)
                self.assertEqual(embedded,(ROOT/f'src/rd_ultrasound_ml/{module}.py').read_text())
            for cell in notebook['cells']:
                if cell['cell_type']=='code':compile(''.join(cell['source']),name,'exec')
            if name.endswith('training'):
                self.assertFalse(any(c['id'] in {'kaggle-download-helper','kaggle-cpu-prepare'} for c in notebook['cells']))


if __name__=='__main__':
    unittest.main()
