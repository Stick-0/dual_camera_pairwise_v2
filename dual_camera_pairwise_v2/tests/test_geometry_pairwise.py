"""End-to-end solver gates with actual original FK; GT only creates/evaluates data."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pairwise.runtime import configure_runtime, REPO_ROOT, require_original_sources
configure_runtime()
from pairwise.geometry import pose,project,transform,pose_error,inverse,normalized
from pairwise.solver import solve_pair

require_original_sources(['ur5e_capture/geometry.py'])
spec=importlib.util.spec_from_file_location('_v2_synthetic_original_fk',REPO_ROOT/'ur5e_capture/geometry.py')
fk=importlib.util.module_from_spec(spec); spec.loader.exec_module(fk)


def fixture(n=24, baseline=.35, noise=0., ambiguous=False):
    rng=np.random.default_rng(1001)
    cameras={
        'camera1':{'K':[[590.,0,320.],[0,605.,240.],[0,0,1.]],'D':[.08,-.03,.001,-.002,0.], 'width':640,'height':480,'pixel_space':'native_distorted'},
        'camera2':{'K':[[630.,0,317.],[0,598.,235.],[0,0,1.]],'D':[-.06,.02,-.001,.001,0.], 'width':640,'height':480,'pixel_space':'native_distorted'}}
    A=pose([.2,-.15,.1,.15,-.1,2.])
    G=pose([.03,.18,-.02,-baseline,.02 if baseline>.01 else 0.,.04 if baseline>.01 else 0.])
    q1=np.array([-.3,-1.,1.2,-.8,.5,.1]); q2=q1+np.array([.02,-.03,.01,.04,-.02,.01])
    truth={'camera1':A,'camera2':G@A}
    observations={}
    for cid,q in zip(('camera1','camera2'),(q1,q2)):
        X=fk.ur5e_keypoints_from_joints(q)
        u=project(transform(truth[cid],X),cameras[cid])+rng.normal(0,noise,(7,2))
        hypotheses=[{'uv':[point.tolist()],'original_scores':[1.]} for point in u]
        observations[cid]={'keypoints_base_m':X,'keypoints_uv':u,'covariance_uv':np.tile(np.eye(2)*4,(7,1,1)),
            'valid_keypoints':np.ones(7,bool),'q_rad':q,'hypotheses':hypotheses}
    if ambiguous:
        # Two coherent full-robot pose hypotheses in C1. Incorrect rotation cannot
        # be removed by the single unknown stereo baseline scale.
        wrong=pose([.30,-.23,.20,.13,-.06,2.06])
        wronguv=project(transform(wrong,observations['camera1']['keypoints_base_m']),cameras['camera1'])
        gooduv=observations['camera1']['keypoints_uv'].copy()
        observations['camera1']['keypoints_uv']=.6*wronguv+.4*gooduv
        observations['camera1']['hypotheses']=[{'uv':[bad.tolist(),good.tolist()],'original_scores':[.6,.4]} for bad,good in zip(wronguv,gooduv)]
    P=np.column_stack((rng.uniform(-.65,.65,n),rng.uniform(-.4,.4,n),rng.uniform(1.5,3.,n)))
    points1=project(P,cameras['camera1'])+rng.normal(0,noise,(n,2))
    points2=project(transform(G,P),cameras['camera2'])+rng.normal(0,noise,(n,2))
    matches={'points_c1_native':points1,'points_c2_native':points2,'match_score':np.ones(n),'valid_mask':np.ones(n,bool)}
    config={'solver':{'max_nfev':40,'max_rgb_matches':24,'max_joint_starts':2,'max_pose_starts':3},
            'candidates':{'kernel_std_px':2.,'outlier_probability':.05,'max_em_iterations':6}}
    return observations,cameras,matches,truth,G,config


class PairGeometryTests(unittest.TestCase):
    def test_noise_free_actual_fk_different_q_and_distortion(self):
        obs,cameras,matches,truth,G,config=fixture()
        result=solve_pair('synthetic_clean',obs,cameras,matches,config,['mono_mean','dual_mean','dual_no_cross'])
        self.assertEqual(result['dual_mean']['status'],'stereo_refined',result['dual_mean']['quality'])
        for cid in truth:
            error=pose_error(result['dual_mean']['poses'][cid],truth[cid])
            self.assertLess(error['translation_m'],2e-5)
            self.assertLess(error['rotation_deg'],.002)
            np.testing.assert_allclose(result['dual_no_cross']['poses'][cid],result['mono_mean']['poses'][cid],atol=1e-12)
        np.testing.assert_allclose(result['dual_mean']['stereo_pose']['T_C2_C1'],G,atol=2e-5)
        self.assertTrue(result['dual_mean']['epipolar_gating_skipped'])
        self.assertIsNone(result['dual_no_cross']['stereo_pose'])

    def test_ambiguous_candidates_resolved_by_actual_cross_view(self):
        obs,cameras,matches,truth,G,config=fixture(ambiguous=True)
        config['solver']['max_joint_starts']=3
        result=solve_pair('synthetic_ambiguous',obs,cameras,matches,config,['mono_candidates','dual_candidates'])
        self.assertEqual(result['dual_candidates']['status'],'stereo_refined',result['dual_candidates']['quality'])
        mono=pose_error(result['mono_candidates']['poses']['camera1'],truth['camera1'])
        dual=pose_error(result['dual_candidates']['poses']['camera1'],truth['camera1'])
        self.assertGreater(mono['rotation_deg'],1.)
        self.assertLess(dual['rotation_deg'],mono['rotation_deg']*.25)
        p=result['dual_candidates']['posterior']['camera1']
        self.assertGreater(sum(x['responsibilities'][1] for x in p),6.)

    def test_small_baseline_and_bad_matches_never_fake_stereo(self):
        for small in (True,False):
            obs,cameras,matches,truth,G,config=fixture(baseline=1e-5 if small else .35)
            if not small:
                matches['points_c2_native']=matches['points_c2_native'][::-1].copy()
            result=solve_pair('degenerate_'+str(small),obs,cameras,matches,config,['dual_mean'])['dual_mean']
            self.assertNotEqual(result['status'],'stereo_refined')
            self.assertIsNone(result['stereo_pose'])
            self.assertTrue(all(v is None for v in result['poses'].values()))

    def test_one_camera_pnp_failure_recovered_by_metric_scale_route(self):
        obs,cameras,matches,truth,G,config=fixture()
        obs['camera1']['valid_keypoints'][3:]=False
        result=solve_pair('one_sided',obs,cameras,matches,config,['mono_mean','dual_mean'])
        self.assertIsNone(result['mono_mean']['poses']['camera1'])
        dual=result['dual_mean']
        self.assertEqual(dual['status'],'stereo_refined',dual['quality'])
        self.assertEqual(dual['quality']['initialization_route'],'rgb_essential_metric_scale')
        self.assertTrue(any(d.get('one_sided_monocular_initialization') for d in dual['quality']['initialization_diagnostics']))
        for cid in truth:
            self.assertLess(pose_error(dual['poses'][cid],truth[cid])['translation_m'],2e-5)

    def test_noisy_outlier_effective_weights_and_nuisance_rank(self):
        obs,cameras,matches,truth,G,config=fixture(noise=.35)
        matches['points_c2_native'][0]+=[90,-80]
        dual=solve_pair('noisy',obs,cameras,matches,config,['dual_mean'])['dual_mean']
        self.assertEqual(dual['status'],'stereo_refined',dual['quality'])
        self.assertLess(dual['quality']['effective_rgb_correspondences'],len(matches['valid_mask']))
        self.assertEqual(dual['quality']['reduced_rank'],12)
        self.assertEqual(dual['quality']['nuisance_policy'],'per_landmark_rank_revealing_elimination')
        for cid in truth:
            self.assertLess(pose_error(dual['poses'][cid],truth[cid])['translation_m'],.05)

    def test_identical_image_provenance_cannot_pass_stereo(self):
        obs,cameras,matches,_,_,config=fixture()
        matches['provenance']={'image_sha256':['same','same']}
        result=solve_pair('copied_rgb',obs,cameras,matches,config,['dual_mean'])['dual_mean']
        self.assertEqual(result['reason'],'identical_image_pair')
        self.assertIsNone(result['stereo_pose'])

    def test_normalized_pixels_remove_each_camera_distortion_once(self):
        obs,cameras,matches,truth,G,config=fixture()
        for cid in cameras:
            X=transform(truth[cid],obs[cid]['keypoints_base_m'])
            expected=X[:,:2]/X[:,2:]
            np.testing.assert_allclose(normalized(obs[cid]['keypoints_uv'],cameras[cid]),expected,atol=2e-8)

    def test_missing_matches_and_candidate_methods_do_not_block_monocular(self):
        obs,cameras,_,_,_,config=fixture()
        obs['camera1'].pop('hypotheses')
        result=solve_pair('missing',obs,cameras,None,config)
        self.assertEqual(result['mono_mean']['status'],'monocular_refined')
        self.assertEqual(result['dual_mean']['status'],'geometry_failed')
        self.assertIsNone(result['dual_mean']['stereo_pose'])
        self.assertIsNotNone(result['dual_mean']['fallback_pose']['camera1'])
        self.assertEqual(result['mono_candidates']['mono_quality']['camera1']['status'],'candidates_unavailable')

    def test_no_hidden_temporal_reference_and_raw_immutable(self):
        obs,cameras,matches,truth,G,config=fixture(n=12)
        before=copy.deepcopy(obs)
        first=solve_pair('same_pair',obs,cameras,matches,config,['mono_mean','mono_candidates','dual_mean','dual_candidates'])
        # Arbitrary legacy and prohibited fields must be ignored, even if malformed.
        changed=copy.deepcopy(obs)
        for cid in changed:
            changed[cid]['legacy_pose']={'pose':'poison'}
            changed[cid]['reference_pose']=np.eye(4)*999
            changed[cid]['projected_location']=np.ones((7,2))*99999
        solve_pair('other_pair',obs,cameras,matches,config,['mono_mean','dual_mean'])
        last=solve_pair('same_pair',changed,cameras,matches,config,['mono_mean','mono_candidates','dual_mean','dual_candidates'])
        for method in first:
            for cid in truth:
                np.testing.assert_array_equal(first[method]['poses'][cid],last[method]['poses'][cid])
        for cid in obs:
            for key in ('keypoints_uv','covariance_uv','valid_keypoints','keypoints_base_m','q_rad'):
                np.testing.assert_array_equal(obs[cid][key],before[cid][key])


if __name__=='__main__':
    unittest.main()
