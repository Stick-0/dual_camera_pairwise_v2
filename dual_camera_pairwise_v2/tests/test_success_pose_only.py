"""Population-policy tests: failed poses, missing points, empty and paired sets."""
import unittest
import numpy as np
from evaluate_success_pose_only import aggregate_successful, paired_common, eligible


def observation(pid, method='dual_candidates', add=20., success=True):
    dual = method.startswith('dual_')
    row = dict(pair_id=pid, camera='camera1', method=method,
               pose_success=int(success), stereo_refined=int(success and dual),
               status='stereo_refined' if dual and success else 'monocular_refined',
               add_mm=add if success else None, translation_error_mm=add if success else None,
               rotation_error_deg=1. if success else None, pair_runtime_ms=5.,
               relative_rotation_error_deg=None, translation_direction_error_deg=None,
               baseline_length_error_mm=None)
    for track in ('raw_voting', 'candidate_selected', 'pose_projected'):
        row['_' + track + '_errors'] = np.array([1., 1., np.inf, np.nan, np.nan, np.nan, np.nan])
    return row


class SuccessPosePolicyTest(unittest.TestCase):
    def test_failed_pose_excluded_but_missing_keypoint_remains_failure(self):
        row, _ = aggregate_successful([observation('a'), observation('b', success=False)], 'dual_candidates', 'camera1')
        self.assertEqual((row['n_attempted'], row['n_evaluated']), (2, 1))
        self.assertEqual(row['ADD_success_rate'], 1.)
        self.assertEqual(row['pose_success_rate'], .5)
        self.assertEqual(row['candidate_selected_reference_points'], 3)
        self.assertAlmostEqual(row['pose_projected_PCK'], 2/3)

    def test_solver_success_is_not_ADD_threshold_or_fallback(self):
        bad_but_accepted = observation('a', add=200.)
        self.assertTrue(eligible(bad_but_accepted))
        fallback = observation('b')
        fallback.update(status='fallback', stereo_refined=0)
        self.assertFalse(eligible(fallback))
        row, _ = aggregate_successful([bad_but_accepted, fallback], 'dual_candidates', 'camera1')
        self.assertEqual(row['n_evaluated'], 1)
        self.assertEqual(row['ADD_success_rate'], 0.)
        self.assertEqual(row['conditional_ADD_mean'], 200.)

    def test_no_success_is_unavailable_not_zero_accuracy(self):
        row, _ = aggregate_successful([observation('a', success=False)], 'dual_candidates', 'camera1')
        self.assertEqual(row['n_evaluated'], 0)
        self.assertIsNone(row['ADD_success_rate'])
        self.assertIsNone(row['ADD_AUC'])
        self.assertIsNone(row['pose_projected_PCK'])

    def test_paired_thresholds_use_same_successful_ids(self):
        rows = [observation('a', add=20.), observation('b', success=False),
                observation('a', method='mono_candidates', add=80.),
                observation('b', method='mono_candidates', add=10.)]
        pair = paired_common(rows, 'dual_candidates', 'camera1')
        self.assertEqual(pair['n_common_success'], 1)
        self.assertEqual(pair['paired_ADD_delta_mean_mm'], -60.)
        self.assertEqual(pair['method_ADD_at_50mm'], 1.)
        self.assertEqual(pair['baseline_ADD_at_50mm'], 0.)


if __name__ == '__main__':
    unittest.main()
