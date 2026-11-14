import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pairwise.runtime import configure_runtime,EXT_ROOT,identity,sha256,write_json
configure_runtime()
from export_infer_csv import export_csv,CSV_COLUMNS


class InferCsvExportTests(unittest.TestCase):
    def fixture(self,root,false_success=False):
        raw=root/'raw.npz'
        uv=np.column_stack((np.arange(7)+10.,np.arange(7)+20.))
        valid=np.array([True,False,True,True,True,True,True])
        np.savez(raw,keypoints_uv=uv,covariance_uv=np.tile(np.eye(2),(7,1,1)),
                 valid_keypoints=valid,q_rad=np.zeros(6),keypoints_base_m=np.zeros((7,3)),
                 foreground_exclusion_mask=np.zeros((480,640),bool))
        obs={'frames':{},'cameras':{}}
        pair={'pair_id':'000000','split':'dev','cameras':{}}
        for cid in ('camera1','camera2'):
            image=f'/read_only_dataset/{cid}/000000.png'
            obs['frames'][cid+'/000000']={'image_path':image,'image_sha256':'unused-read-only-fixture',
                'arrays_path':str(raw),'arrays_sha256':sha256(raw),'hypotheses':[{'uv':[v.tolist()],'original_scores':[1.]} for v in uv]}
            pair['cameras'][cid]={'image_path':image,'image_sha256':'unused-read-only-fixture','q_rad':[0.]*6,
                'observation_key':cid+'/000000','K':[[600.,0,320.],[0,600.,240.],[0,0,1.]],'D':[0.]*5,
                'width':640,'height':480,'pixel_space':'native_distorted'}
        obs['identity']=identity(obs);obs_path=root/'observations.json';write_json(obs_path,obs)
        pairs=root/'pairs.jsonl';pairs.write_text(json.dumps(pair)+'\n')
        T=np.eye(4);T[2,3]=1.
        mono={'status':'monocular_refined','poses':{c:T for c in pair['cameras']},'stereo_pose':None,'fallback_pose':None}
        dual={'status':'stereo_refined' if false_success else 'geometry_failed','reason':'test_cross_failure',
              'poses':{c:None for c in pair['cameras']},'stereo_pose':None,'fallback_pose':{c:T for c in pair['cameras']},
              'quality':{'actual_rgb_correspondences_used':0},
              'candidate_selected_uv':{c:uv for c in pair['cameras']},
              'candidate_selected_valid':{c:[True]*7 for c in pair['cameras']}}
        run=root/'run';run.mkdir()
        result=run/'result.json';write_json(result,{'pair_id':'000000','split':'dev','config_identity':'test',
            'methods':{'mono_mean':mono,'dual_candidates':dual}})
        write_json(run/'manifest.json',{'completed':True,'pairs_sha256':sha256(pairs),'config_identity':'test',
            'observations_identity':obs['identity'],'methods':['mono_mean','dual_candidates'],
            'results':[{'pair_id':'000000','result_path':str(result),'sha256':sha256(result)}]})
        return run,pairs,obs_path,raw

    def test_column_order_validity_units_and_failed_stereo_no_fallback_mix(self):
        with tempfile.TemporaryDirectory(dir=EXT_ROOT/'tmp') as temp:
            root=Path(temp);run,pairs,obs,raw=self.fixture(root);before=sha256(raw)
            result=export_csv(run,pairs,obs,root/'export')
            self.assertEqual(before,sha256(raw))
            with (root/'export/mono_mean_camera1_raw_voting.csv').open() as stream:
                rows=list(csv.reader(stream))
            self.assertEqual(rows[0],CSV_COLUMNS)
            self.assertEqual(rows[1][1],'10.0');self.assertEqual(rows[1][2],'')
            self.assertEqual(rows[1][8],'20.0');self.assertEqual(rows[1][9],'')
            by_name=dict(zip(rows[0],rows[1]))
            self.assertEqual([float(by_name[k]) for k in ('quat_x','quat_y','quat_z','quat_w')],[0.,0.,0.,1.])
            self.assertEqual(float(by_name['tvec_z']),1.)
            self.assertEqual(by_name['pnp_success'],'1')
            for stage in ('raw_voting','candidate_selected','pose_projected'):
                with (root/f'export/dual_candidates_camera1_{stage}.csv').open() as stream:
                    row=next(csv.DictReader(stream))
                self.assertEqual(row['pnp_success'],'0')
                self.assertTrue(all(row[k]=='' for k in ('quat_x','quat_y','quat_z','quat_w','tvec_x','tvec_y','tvec_z','reproj_mean')))
                if stage!='raw_voting':self.assertTrue(all(row[k]=='' for k in CSV_COLUMNS[1:15]))
            with (root/'export/row_statuses.csv').open() as stream:
                statuses=list(csv.DictReader(stream))
            failures=[r for r in statuses if r['method']=='dual_candidates']
            self.assertTrue(all(r['fallback_available']=='1' and r['stereo_refined']=='0' for r in failures))
            self.assertFalse(result['reference_accessed'])

    def test_false_stereo_success_rejected_instead_of_exported(self):
        with tempfile.TemporaryDirectory(dir=EXT_ROOT/'tmp') as temp:
            root=Path(temp);run,pairs,obs,_=self.fixture(root,false_success=True)
            with self.assertRaisesRegex(ValueError,'False stereo success'):
                export_csv(run,pairs,obs,root/'export')


if __name__=='__main__':unittest.main()
