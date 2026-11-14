"""Population regression: full dataset includes both old split labels."""
import sys
import unittest
from pathlib import Path
from evaluate_all_success import EXT, METHODS, summarize_all

sys.path.insert(0, str(EXT / 'tests'))
from test_success_pose_only import SuccessPosePolicyTest, observation


class FullPopulationTest(unittest.TestCase):
    def test_no_temporal_subset_and_same_pair_policy(self):
        rows = []
        for method in METHODS:
            for index, split in enumerate(('dev', 'final')):
                for camera in ('camera1', 'camera2'):
                    row = observation(str(index), method=method,
                        success=not (method == 'dual_candidates' and index == 0))
                    row.update(camera=camera, split=split)
                    if method == 'dual_no_cross':
                        row.update(stereo_refined=0, status='no_cross_refined')
                    rows.append(row)
        summaries, curves, paired, checks = summarize_all(rows, 'roma_v2', 'test_two_pairs')
        self.assertEqual(len(summaries), 15)
        self.assertTrue(all(r['split'] == 'all' for r in summaries))
        result = next(r for r in summaries if r['method'] == 'dual_candidates' and r['camera'] == 'pooled')
        self.assertEqual((result['n_attempted'], result['n_evaluated']), (4, 2))
        self.assertEqual(result['n_attempted_unique_pairs'], 2)
        common = next(r for r in paired if r['method'] == 'dual_candidates' and r['camera'] == 'pooled')
        self.assertEqual(common['n_common_success'], 2)
        self.assertTrue(all(r['passed'] for r in checks))


if __name__ == '__main__':
    unittest.main(verbosity=2)
