"""Meaningful denominator, masking, temporal resampling and reference tests."""
import unittest
import numpy as np
from evaluate import (auc_scores,distribution,comparison_statistics,
    bootstrap_block_counts,bootstrap_ratios,pck_errors,load_arrays,METHODS)


class EvaluationTests(unittest.TestCase):
    def test_failed_poses_stay_in_threshold_denominator(self):
        summary,curve=distribution([0.,50.,np.inf,np.inf])
        self.assertEqual(summary['n_total'],4)
        self.assertEqual(summary['n_valid'],2)
        self.assertEqual(summary['ADD50'],.25)
        self.assertAlmostEqual(summary['ADD_AUC'],(.975+.475)/4)
        self.assertEqual(curve[0],0)
        self.assertEqual(curve[-1],.5)
        self.assertTrue(np.allclose(auc_scores([0.,50.,np.inf]),[.975,.475,0]))

    def test_paired_difference_uses_intersection_not_marginal_means(self):
        errors=np.full((3,4,2),np.inf)
        errors[:,0]=[[10,10],[100,100],[5,5]]
        errors[:,1]=[[20,20],[np.inf,np.inf],[3,3]]
        errors[:,2]=errors[:,0];errors[:,3]=errors[:,1]
        rows,_,_=comparison_statistics(errors,np.ones(3,bool),[0,1])
        r=rows[0]
        self.assertEqual(r['n_denominator'],6)
        self.assertEqual(r['n_common_valid'],4)
        self.assertEqual(r['paired_ADD_delta_mm'],4.)
        self.assertNotEqual(r['paired_ADD_delta_mm'],11.5-(10+100+5)/3)
        self.assertAlmostEqual(r['ADD50_delta'],0.)

    def test_time_blocks_resample_before_eligibility_and_validity(self):
        ts=np.array([0.,.1,1.,1.1,2.,2.1])
        assignment,counts,meta=bootstrap_block_counts(ts,1.,20,0)
        counts=np.vstack([counts,[0,3,0]])  # a valid draw containing no eligible pair
        np.testing.assert_array_equal(assignment,[0,0,1,1,2,2])
        numerator=np.array([0.,0.,0.,0.,100.,0.])
        denominator=np.array([1.,0.,0.,0.,1.,0.])
        actual=bootstrap_ratios(numerator,denominator,assignment,counts)[:,0]
        for i,row in enumerate(counts):
            indices=np.concatenate([np.tile(np.flatnonzero(assignment==b),n) for b,n in enumerate(row)])
            den=denominator[indices].sum()
            expected=numerator[indices].sum()/den if den else np.nan
            self.assertTrue(np.isclose(actual[i],expected,equal_nan=True))
        # Original middle time block remains possible even with no eligible rows.
        self.assertEqual(meta['n_blocks'],3)
        self.assertTrue(np.isnan(actual).any())

    def test_same_block_draws_are_shared_by_all_methods(self):
        ts=np.arange(40)*.1
        a,c,_=bootstrap_block_counts(ts,1.,50,0)
        nums=np.stack([np.arange(40),np.arange(40)+7],axis=1)
        values=bootstrap_ratios(nums,np.ones_like(nums),a,c)
        np.testing.assert_allclose(values[:,1]-values[:,0],7.)
        a2,c2,_=bootstrap_block_counts(ts,1.,50,0)
        np.testing.assert_array_equal(a,a2);np.testing.assert_array_equal(c,c2)

    def test_reference_outside_is_excluded_but_missing_prediction_fails(self):
        ref=np.array([[1.,1.]]*7);ref[0]=[-1,0]
        errors=pck_errors(None,ref,{'width':10,'height':10})
        self.assertTrue(np.isnan(errors[0]));self.assertTrue(np.isinf(errors[1:]).all())

    def test_reference_changes_metrics_not_estimate_and_gate_masks_pose(self):
        T=np.eye(4);T[2,3]=2
        cam=dict(timestamp_s=1.,keypoints_base_m=np.zeros((7,3)).tolist(),K=np.eye(3).tolist(),D=[],width=640,height=480)
        pair={'pair_id':'p','cameras':{'camera1':cam,'camera2':cam}}
        refcam=dict(T_camera_from_base=T.tolist(),keypoints_base_m=cam['keypoints_base_m'],projected_keypoints_uv=np.zeros((7,2)).tolist())
        ref={'pair_id':'p','cameras':{'camera1':dict(refcam),'camera2':dict(refcam)}}
        methods={m:dict(poses={c:T.tolist() for c in ('camera1','camera2')},common_valid=True,final_valid=m!='R1',raw_inputs={}) for m in METHODS}
        result={'pair_id':'p','pre_eligible':True,'methods':methods}
        original=repr(result)
        a=load_arrays({'p':result},{'p':pair},{'p':ref})
        self.assertTrue((a['add']==0).all());self.assertFalse(a['valid']['final'][0,1].any())
        changed=T.copy();changed[0,3]=.1
        ref['cameras']['camera1']['T_camera_from_base']=changed.tolist()
        b=load_arrays({'p':result},{'p':pair},{'p':ref})
        self.assertTrue(np.allclose(b['add'][:,:,0],100))
        self.assertEqual(repr(result),original)


if __name__=='__main__':unittest.main()
