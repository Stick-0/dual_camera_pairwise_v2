"""Check failed denominators, fallback separation, and reference-only metrics."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pairwise.evaluation import (
    METHODS, _pck_errors, _relative_metrics, _row_metrics, evaluate_pairs, prepare_reference,
)
from pairwise.runtime import EXT_ROOT, configure_runtime, jsonable, write_json

configure_runtime()


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='evaluation_test_', dir=EXT_ROOT / 'tmp'))
        self.camera = {'K': [[100., 0, 50], [0, 100, 50], [0, 0, 1]], 'D': [0, 0, 0, 0, 0],
                       'width': 100, 'height': 100}
        self.xyz = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1],
                             [.1, .1, 0], [.1, 0, .1], [0, .1, .1]])
        self.pairs = []
        dataset = self.root / 'dataset'
        dataset.mkdir()
        dataset_rows = []
        for index in range(3):
            pid = f'{index:06d}'
            pair = {'pair_id': pid, 'split': 'final', 'cameras': {}}
            source = {'sample_id': pid, 'cameras': {}}
            for camera_index, cid in enumerate(('camera1', 'camera2')):
                T = np.eye(4)
                T[:3, 3] = [camera_index * .2, 0, 2]
                uv = cv2.projectPoints(self.xyz, np.zeros(3), T[:3, 3],
                                        np.array(self.camera['K']), np.zeros(5))[0].reshape(7, 2)
                record = dict(self.camera, keypoints_base_m=self.xyz, image_path=str(self.root / f'{pid}_{cid}.png'))
                pair['cameras'][cid] = record
                cv2.imwrite(record['image_path'], np.full((100, 100, 3), 130, np.uint8))
                obj = {'class': 'ur5e', 'location_cameraframe': T[:3, 3],
                       'xyzw_cameraframe': [0, 0, 0, 1],
                       'keypoints': [{'name': f'ur5e_link{k}', 'location': self.xyz[k],
                                      'projected_location': uv[k]} for k in range(7)]}
                path = dataset / f'{pid}_{cid}.json'
                write_json(path, {'objects': [obj]})
                source['cameras'][cid] = {'gt': path.name}
            self.pairs.append(pair)
            dataset_rows.append(source)
        with (dataset / 'frames.jsonl').open('x') as stream:
            stream.write('\n'.join(json.dumps(row) for row in dataset_rows))
        self.reference = self.root / 'reference.jsonl'
        self.reference_info = prepare_reference(self.pairs, dataset, self.reference)
        self.refs = {row['pair_id']: row for row in map(json.loads, self.reference.read_text().splitlines())}

    def test_failure_denominators_fallback_and_paired_effects(self):
        run = self.root / 'run'
        (run / 'results').mkdir(parents=True)
        manifest = []
        # The third expected pair is entirely absent: it must stay in all denominators.
        for pair in self.pairs[:2]:
            pid = pair['pair_id']
            poses = {cid: np.array(ref['T_camera_from_base']) for cid, ref in self.refs[pid]['cameras'].items()}
            observations = {cid: {'keypoints_uv': ref['projected_keypoints_uv'],
                                  'valid_keypoints': [False, True, True, True, True, True, True],
                                  'keypoints_base_m': self.xyz}
                            for cid, ref in self.refs[pid]['cameras'].items()}
            selected = {cid: o['keypoints_uv'] for cid, o in observations.items()}
            methods = {}
            for method in METHODS:
                item = {'poses': copy.deepcopy(poses), 'status': 'monocular_refined',
                        'posterior': {cid: None for cid in poses},
                        'candidate_selected_uv': selected, 'candidate_selected_valid': {cid: [True] * 7 for cid in poses},
                        'timing': {'total_ms': 10, 'initialization_ms': 2, 'ba_ms': 8}}
                if method in ('dual_mean', 'dual_candidates'):
                    item['poses'] = {cid: None for cid in poses}
                    if pid == '000000':
                        shifted = copy.deepcopy(poses)
                        for T in shifted.values():
                            T[0, 3] += .01
                        item.update(status='stereo_refined', stereo_pose=shifted,
                                    quality={'actual_rgb_correspondences_used': 10})
                    else:
                        item.update(status='geometry_failed', reason='few_matches', stereo_pose=None,
                                    fallback_pose=copy.deepcopy(poses),
                                    mono_quality={cid:{'reason':'earlier_mono_diagnostic'} for cid in poses})
                methods[method] = item
            result = {'pair_id': pid, 'methods': methods, 'observations': observations,
                      'timing': {'matcher_ms': 4, 'robot_observations_reused': True}}
            path = run / 'results' / f'{pid}.json'
            write_json(path, result)
            manifest.append({'pair_id': pid, 'result_path': str(path)})
        write_json(run / 'manifest.json', {'results': manifest})
        report = evaluate_pairs(self.pairs, self.reference, run, self.root / 'evaluation')
        summaries = {(r['method'], r['camera']): r for r in report['summary']}
        mono = summaries['mono_candidates', 'pooled']
        dual = summaries['dual_candidates', 'pooled']
        self.assertEqual(mono['n_total'], 6)
        self.assertEqual(mono['n_pose_success'], 4)
        self.assertEqual(dual['n_pose_success'], 2)
        self.assertEqual(dual['n_stereo_refined'], 2)
        self.assertEqual(dual['n_unique_stereo_pairs'], 1)
        self.assertAlmostEqual(dual['ADD_success_rate'], 1 / 3)
        self.assertAlmostEqual(dual['conditional_ADD_mean'], 10.)
        self.assertAlmostEqual(dual['paired_delta_vs_matching_mono'], 10.)
        self.assertAlmostEqual(dual['pair_runtime_ms'], 14.)
        self.assertAlmostEqual(mono['pair_runtime_ms'], 10.)
        self.assertEqual(dual['common_success_count'], 2)
        self.assertAlmostEqual(mono['raw_voting_PCK_at_5px'], 24 / 42)
        self.assertAlmostEqual(dual['candidate_PCK'], 1 / 3)
        fallback = next(r for r in report['operational_fallback_summary']
                        if r['method'] == 'dual_candidates' and r['camera'] == 'pooled')
        self.assertEqual(fallback['n_pose_success'], 2)
        self.assertEqual(fallback['n_stereo_refined'], 0)
        self.assertAlmostEqual(fallback['conditional_ADD_mean'], 0.)
        self.assertEqual(summaries['dual_no_cross', 'pooled']['n_stereo_refined'], 0)
        with Path(report['artifacts']['per_pair_metrics']).open() as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(len(records), 3 * 2 * 5)
        failed = [r for r in records if r['method'] == 'dual_candidates' and r['pair_id'] == '000001']
        self.assertTrue(all(r['add_mm'] == '' and r['pose_success'] == '0' for r in failed))
        self.assertTrue(all(r['reason'] == 'few_matches' for r in failed))
        self.assertTrue(all(r['monocular_initialization_reason'] == 'earlier_mono_diagnostic' for r in failed))
        missing = [r for r in records if r['pair_id'] == '000002']
        self.assertTrue(all(r['status'] == 'not_executed' for r in missing))
        with Path(report['artifacts']['timing']).open() as stream:
            timings = list(csv.DictReader(stream))
        self.assertTrue(all(r['robot_forward_ms'] == '' for r in timings))
        self.assertTrue(all(float(r['pair_runtime_ms']) == 14. for r in timings
                            if r['method'] in ('dual_mean', 'dual_candidates')))
        self.assertTrue(all(float(r['solver_total_ms']) == 10. for r in timings))
        overlays = report['artifacts']['representative_overlays']
        self.assertTrue(any(e['category'] == 'worsened' and e['path'] for e in overlays))
        self.assertTrue(all(e['path'] is None for e in overlays if e['category'] == 'improved'))

    def test_reference_geometry_and_metadata_are_explicit(self):
        self.assertFalse(self.reference_info['reference_is_independent_gt'])
        self.assertAlmostEqual(self.reference_info['max_reference_projection_difference_px'], 0.)
        for record in self.refs.values():
            self.assertEqual(record['role'], 'evaluation_only')
            self.assertAlmostEqual(record['T_C2_C1_reference'][0][3], .2)
        # No reference pose or projected label was added to an estimation input.
        for pair in self.pairs:
            for camera in pair['cameras'].values():
                self.assertNotIn('T_camera_from_base', camera)
                self.assertNotIn('projected_keypoints_uv', camera)

    def test_pck_oob_prediction_is_not_removed_and_invalid_stays_failed(self):
        reference = np.tile([99., 50.], (7, 1))
        prediction = np.tile([101., 50.], (7, 1))
        valid = [False] + [True] * 6
        errors = _pck_errors(prediction, reference, self.camera, valid)
        self.assertTrue(np.isinf(errors[0]))
        np.testing.assert_allclose(errors[1:], 2.)
        self.assertEqual(np.mean(errors < 2), 0.)
        self.assertAlmostEqual(np.mean(errors < 5), 6 / 7)
        reference[0] = [100, 50]
        self.assertTrue(np.isnan(_pck_errors(prediction, reference, self.camera)[0]))

    def test_relative_errors_keep_translation_sign_and_metric_scale(self):
        reference, estimate = np.eye(4), np.eye(4)
        reference[:3, 3] = [.2, 0, 0]
        estimate[:3, 3] = [-.3, 0, 0]
        estimate[:3, :3] = Rotation.from_euler('z', 15, degrees=True).as_matrix()
        errors = _relative_metrics(estimate, reference)
        self.assertAlmostEqual(errors['relative_rotation_error_deg'], 15)
        self.assertAlmostEqual(errors['translation_direction_error_deg'], 180)
        self.assertAlmostEqual(errors['baseline_length_error_mm'], 100)

    def test_success_flag_without_actual_rgb_factor_is_rejected(self):
        pair = self.pairs[0]
        reference = self.refs[pair['pair_id']]
        poses = {cid: data['T_camera_from_base'] for cid, data in reference['cameras'].items()}
        result = {'methods': {'dual_candidates': {'status': 'stereo_refined',
                   'stereo_pose': poses, 'quality': {'actual_rgb_correspondences_used': 0}}}}
        with self.assertRaisesRegex(ValueError, 'actual RGB factor'):
            _row_metrics(pair, 'camera1', 'dual_candidates', result, reference, {})


if __name__ == '__main__':
    unittest.main()
