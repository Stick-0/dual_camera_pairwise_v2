import unittest
import numpy as np

from pairwise.matching import filter_matches


class MatchFilteringTests(unittest.TestCase):
    def test_masks_are_current_endpoints_and_inputs_preserved(self):
        p1=np.array([[10.,10.],[30.,30.],[50.,50.],[70.,70.],[90.,90.]])
        p2=p1.copy(); scores=np.ones(5); cycle=np.zeros(5)
        m1=np.zeros((100,100),bool);m2=m1.copy();m2[30,30]=True
        before=[x.copy() for x in (p1,p2,m1,m2)]
        valid,reason,counts,_=filter_matches(p1,p2,scores,cycle,[m1,m2],[(100,100)]*2,
            {'mask_dilation_px':0,'grid_cell_px':8,'min_spacing_px':0})
        self.assertEqual(valid.tolist(),[True,False,True,True,True])
        self.assertEqual(reason[1],'current_robot_foreground')
        for old,new in zip(before,(p1,p2,m1,m2)):np.testing.assert_array_equal(old,new)

    def test_cycle_score_bounds_and_dedup_reasons(self):
        p=np.array([[1.,1.],[2.,2.],[20.,20.],[40.,40.],[60.,60.],[-1.,1.]])
        scores=np.array([1.,.9,.1,.8,.7,.9]); cycle=np.array([0.,0.,0.,4.,0.,0.])
        empty=np.zeros((80,80),bool)
        valid,reasons,counts,_=filter_matches(p,p,scores,cycle,[empty,empty],[(80,80)]*2,
            {'mask_dilation_px':0,'max_spatially_selected_matches':1})
        self.assertEqual(valid.sum(),1)
        self.assertEqual(reasons.tolist(),['accepted','duplicate_grid_cell','low_score',
            'bidirectional_inconsistent','selection_budget','outside_image'])
        self.assertEqual(counts['after_bidirectional_inconsistent'],3)

    def test_no_matches_and_dilation(self):
        mask=np.zeros((20,20),bool);mask[10,10]=True
        valid,_,_,_=filter_matches(np.empty((0,2)),np.empty((0,2)),np.empty(0),np.empty(0),
            [mask,mask],[(20,20)]*2)
        self.assertEqual(valid.shape,(0,))
        p=np.array([[12.,10.]])
        valid,reason,_,_=filter_matches(p,p,[1.],[0.],[mask,mask],[(20,20)]*2,{'mask_dilation_px':2})
        self.assertFalse(valid[0]);self.assertEqual(reason[0],'current_robot_foreground')


if __name__=='__main__':unittest.main()
