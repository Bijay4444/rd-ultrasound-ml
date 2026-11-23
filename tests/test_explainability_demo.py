"""Portable viewer extension integrity and fixed-panel alignment (synthetic files)."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_explainability_demo as demo


class ExplanationDemoTests(unittest.TestCase):
    def fixture(self,root,models,settings=None):
        plan=json.loads((ROOT/'configs/r3d18_gradcam_validation_001.json').read_text())
        cases=plan['selection']['cases']
        files={}
        def add(name,value):files[name]=json.dumps(value).encode()
        pairs=[{'model':m,'clip_id':c['clip_id'],'status':'completed'} for m in models for c in cases]
        add('status.json',{'status':'completed','models':{m:{'status':'completed'} for m in models},'cases':pairs})
        add('selection_manifest.json',plan['selection'])
        add('settings.json',settings or {'seed':42,'threshold':.5})
        add('runtime.json',{'fixture':True})
        add('case_results.json',[{**c,'model':m,'outcome':'TP','fp32_probability':.75} for m in models for c in cases])
        files['sources/video_explainability.py']=b'synthetic algorithm identity'
        for m in models:
            add(m+'/freeze.json',{'threshold':.5,'selected_epoch':{'unetpp':48,'r2plus1d18':15,'r3d18':18}[m]})
            for c in cases:
                key=m+'/'+c['clip_id']+'/'
                add(key+'occlusion.json',{'groups':{},'constant_map':False})
                add(key+'randomization.json',[])
                for name in ('panel.png','temporal.png','rd_overlay.gif','non_rd_overlay.gif'):
                    files[key+name]=('synthetic media '+key+name).encode()
        add('artifact_sha256.json',{n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
        archive=Path(root)/(''.join(models)+'.zip')
        with zipfile.ZipFile(archive,'w') as z:
            for n,b in files.items():z.writestr('run/'+n,b)
        return archive,plan

    def test_combines_three_models_without_replacing_historical_assets(self):
        with TemporaryDirectory() as d:
            reference,plan=self.fixture(d,['unetpp','r2plus1d18'])
            extension,_=self.fixture(d,['r3d18'])
            out=Path(d)/'viewer'
            with patch.object(demo,'file_sha',side_effect=lambda p:plan['reference_archive_sha256'] if p==reference else 'synthetic-extension'):
                demo.build(reference,out,[extension])
            provenance=json.loads((out/'provenance.json').read_text())
            self.assertEqual(provenance['models'],['unetpp','r2plus1d18','r3d18'])
            self.assertTrue(provenance['reused_historical_explanations'])
            self.assertEqual(len(list((out/'assets').rglob('*.png'))),48)
            self.assertEqual(len(list((out/'assets').rglob('*.gif'))),48)
            self.assertIn('R3D-18',(out/'index.html').read_text())
            self.assertIn('\n## R3D-18',(out/'speaker-notes.md').read_text())
            with zipfile.ZipFile(reference) as z:
                self.assertEqual((out/'assets/unetpp/405744_00014/panel.png').read_bytes(),
                                 z.read('run/unetpp/405744_00014/panel.png'))

    def test_rejects_method_drift_before_writing_viewer(self):
        with TemporaryDirectory() as d:
            reference,_=self.fixture(d,['unetpp','r2plus1d18'])
            extension,_=self.fixture(d,['r3d18'],{'seed':43,'threshold':.5})
            with self.assertRaisesRegex(ValueError,'method'):
                demo.build(reference,Path(d)/'viewer',[extension])
            self.assertFalse((Path(d)/'viewer').exists())

    def test_corrupt_archive_fails_before_media_copy(self):
        with TemporaryDirectory() as d:
            archive,_=self.fixture(d,['unetpp','r2plus1d18'])
            with zipfile.ZipFile(archive,'a') as z:z.writestr('run/unlisted.txt',b'changed')
            with self.assertRaisesRegex(ValueError,'membership'):
                demo.build(archive,Path(d)/'viewer')
            self.assertFalse((Path(d)/'viewer').exists())


if __name__=='__main__':unittest.main()
