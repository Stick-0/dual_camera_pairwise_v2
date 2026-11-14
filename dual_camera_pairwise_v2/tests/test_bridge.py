import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pairwise.bridge import ObservationStore, ARRAY_KEYS, load_config
from pairwise.runtime import EXT_ROOT, REPO_ROOT, identity, sha256, output_path, write_json


class BridgeContractTests(unittest.TestCase):
    def test_old_extension_is_never_an_output(self):
        with self.assertRaises(ValueError):
            output_path(REPO_ROOT/'dual_view_refinement/cache/forbidden.npz')
        with self.assertRaises(ValueError):
            output_path(REPO_ROOT/'robotpose/__pycache__/forbidden.pyc')

    def test_cached_temporal_and_reference_fields_are_not_exposed(self):
        with tempfile.TemporaryDirectory(dir=EXT_ROOT/'tmp') as directory:
            root=Path(directory);arrays=root/'raw.npz'
            data=dict(keypoints_uv=np.ones((7,2)),covariance_uv=np.tile(np.eye(2),(7,1,1)),
                      valid_keypoints=np.ones(7,bool),q_rad=np.zeros(6),keypoints_base_m=np.ones((7,3)),
                      foreground_exclusion_mask=np.zeros((8,8),bool))
            np.savez_compressed(arrays,**data)
            row=dict(frame_id='x',image_path='/unchanged.png',image_sha256='image',
                     arrays_path=str(arrays),arrays_sha256=sha256(arrays),
                     observation_stage='raw_voting',hypotheses=[{'uv':[[1,1]],'original_scores':[1]}]*7,
                     legacy_pose={'T':np.eye(4).tolist()},reference={'pose':'POISON'})
            metadata={'frames':{cid+'/x':dict(row,camera_id=cid) for cid in ('camera1','camera2')}}
            metadata['identity']=identity(metadata);path=root/'manifest.json';write_json(path,metadata)
            store=ObservationStore(path)
            pair={'cameras':{cid:dict(observation_key=cid+'/x',image_path='/unchanged.png',image_sha256='image',q_rad=[0]*6) for cid in ('camera1','camera2')}}
            observations=store.pair(pair)
            for observation in observations.values():
                self.assertNotIn('legacy_pose',observation);self.assertNotIn('reference',observation)
                observation['keypoints_uv'][:]=999
            self.assertEqual(store.pair(pair)['camera1']['keypoints_uv'][0,0],1)
            with arrays.open('ab') as file:file.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Raw arrays changed'):store.pair(pair)

    def test_current_pair_policy_cannot_enable_temporal_or_reference(self):
        import yaml
        cfg=load_config(EXT_ROOT/'configs/ur_dual_2596_pairwise.yaml')
        with tempfile.TemporaryDirectory(dir=EXT_ROOT/'tmp') as directory:
            path=Path(directory)/'config.yaml'
            for key in ('temporal_prior','use_reference_for_inference','use_legacy_B_initialization'):
                bad=copy.deepcopy(cfg);bad['estimation'][key]=True;path.write_text(yaml.safe_dump(bad))
                with self.assertRaises(ValueError):load_config(path)


if __name__=='__main__':unittest.main()
