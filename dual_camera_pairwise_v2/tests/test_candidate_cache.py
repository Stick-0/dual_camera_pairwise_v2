"""Physical candidate linkage rejects tampering and preserves numeric evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pairwise.bridge import ARRAY_KEYS, ObservationStore, separate_candidates
from pairwise.runtime import EXT_ROOT, configure_runtime, identity, sha256, write_json

configure_runtime()


class CandidateCacheTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=EXT_ROOT/'tmp')
        self.root = Path(self.directory.name)
        arrays = self.root/'arrays.npz'
        np.savez_compressed(arrays, keypoints_uv=np.ones((7,2)),
            covariance_uv=np.tile(np.eye(2),(7,1,1)), valid_keypoints=np.array([False]+[True]*6),
            q_rad=np.zeros(6), keypoints_base_m=np.zeros((7,3)), foreground_exclusion_mask=np.zeros((8,8),np.uint8))
        hypotheses = [{'uv': [[1.125,2.25], [4.5,8.75]], 'original_scores': [0., 3.125]} for _ in range(7)]
        frames = {cid+'/x': {'camera_id':cid,'frame_id':'x','image_path':'/unchanged.png',
            'image_sha256':'unchanged_rgb','arrays_path':str(arrays),'arrays_sha256':sha256(arrays),
            'hypotheses':copy.deepcopy(hypotheses),'observation_stage':'raw_voting'} for cid in ('camera1','camera2')}
        metadata = {'schema_version':2, 'source_cache':'/original/cache', 'source_cache_identity':'original',
                    'robot_checkpoint_sha256':'frozen_checkpoint', 'pairs_sha256':'frozen_pairs','frames':frames}
        metadata['identity'] = identity(metadata)
        self.source = self.root/'source.json'
        write_json(self.source,metadata)
        self.candidates = self.root/'candidates.json'
        self.observations = self.root/'separated.json'
        self.separated = separate_candidates(metadata, {'candidates':{'kernel_std_px':2.}}, self.candidates, self.source)
        write_json(self.observations,self.separated)
        self.pair = {'cameras': {cid:{'observation_key':cid+'/x', 'image_path':'/unchanged.png',
                       'image_sha256':'unchanged_rgb','q_rad':[0]*6} for cid in ('camera1','camera2')}}

    def tearDown(self):
        self.directory.cleanup()

    def test_separated_loader_is_byte_identical_to_embedded_inputs(self):
        old, new = ObservationStore(self.source), ObservationStore(self.observations)
        self.assertEqual(old.candidate_provenance['storage'],'legacy_embedded_compatibility')
        self.assertEqual(new.candidate_provenance['storage'],'separate_verified_original_hypotheses')
        for row in new.metadata['frames'].values():
            self.assertNotIn('hypotheses',row)
        a,b = old.pair(self.pair),new.pair(self.pair)
        for cid in a:
            for key in ARRAY_KEYS:
                self.assertEqual(a[cid][key].dtype,b[cid][key].dtype)
                self.assertEqual(a[cid][key].tobytes(),b[cid][key].tobytes())
            self.assertEqual(a[cid]['hypotheses'],b[cid]['hypotheses'])
            # Returned posterior/consumer mutations must never affect original cache.
            b[cid]['hypotheses'][0]['uv'][0][0] = 999
        self.assertEqual(new.pair(self.pair)['camera1']['hypotheses'][0]['uv'][0][0],1.125)

    def test_candidate_byte_tampering_rejected_even_when_json_remains_parseable(self):
        with self.candidates.open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError,'Candidate manifest SHA-256 mismatch'):
            ObservationStore(self.observations)

    def test_rehashed_candidate_missing_frame_still_rejected(self):
        candidates = json.loads(self.candidates.read_text())
        candidates['frames'].pop('camera2/x')
        candidates.pop('identity')
        candidates['identity'] = identity(candidates)
        self.candidates.write_text(json.dumps(candidates))
        metadata = copy.deepcopy(self.separated)
        metadata.update(candidate_manifest_sha256=sha256(self.candidates),candidate_manifest_identity=candidates['identity'])
        metadata.pop('identity')
        metadata['identity'] = identity(metadata)
        self.observations.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,'coverage mismatch'):
            ObservationStore(self.observations)


if __name__ == '__main__':
    unittest.main()
