import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pairwise.runtime import configure_runtime
configure_runtime()
from pairwise.candidates import prepare_candidates,responsibilities,candidate_residuals


class CandidateLikelihoodTests(unittest.TestCase):
    def test_duplicate_hypothesis_split_mass_is_identical_information(self):
        camera={'width':640,'height':480}
        z=np.array([[110.,98.]])
        original={'valid_keypoints':[True],'hypotheses':[{'uv':[[100.,100.],[120.,110.]],'original_scores':[.7,.3]}]}
        duplicate={'valid_keypoints':[True],'hypotheses':[{'uv':[[100.,100.],[100.,100.],[120.,110.]],'original_scores':[.35,.35,.3]}]}
        a=prepare_candidates(original,camera); b=prepare_candidates(duplicate,camera)
        pa,na=responsibilities(z,a); pb,nb=responsibilities(z,b)
        self.assertAlmostEqual(na,nb,places=12)
        self.assertAlmostEqual(pa[0]['responsibilities'][0],sum(pb[0]['responsibilities'][:2]),places=12)
        self.assertAlmostEqual(float(np.sum(candidate_residuals(z,a,pa)**2)),float(np.sum(candidate_residuals(z,b,pb)**2)),places=12)
        self.assertAlmostEqual(sum(pa[0]['responsibilities'])+pa[0]['outlier_responsibility'],1.,places=12)

    def test_single_gaussian_quadratic_objective_coefficient(self):
        camera={'width':640,'height':480}
        obs={'valid_keypoints':[True],'hypotheses':[{'uv':[[100.,100.]],'original_scores':[8.]}]}
        prepared=prepare_candidates(obs,camera,{'kernel_std_px':2.,'outlier_probability':0.})
        p,n=responsibilities(np.array([[103.,104.]]),prepared)
        residual=candidate_residuals(np.array([[103.,104.]]),prepared,p)
        self.assertAlmostEqual(n-np.log(2*np.pi*4),.5*np.sum(residual**2),places=12)

    def test_invalid_raw_and_zero_support_never_become_valid(self):
        obs={'valid_keypoints':[False,True],'hypotheses':[{'uv':[[100.,100.]],'original_scores':[1.]},{'uv':[[100.,100.]],'original_scores':[0.]}]}
        p=prepare_candidates(obs,{'width':640,'height':480})
        self.assertFalse(any(m['valid'] for m in p['modes']))
        posts,nll=responsibilities(np.ones((2,2)),p)
        self.assertEqual(nll,0.)
        self.assertTrue(all(v['outlier_responsibility']==1. for v in posts))


if __name__=='__main__':
    unittest.main()
