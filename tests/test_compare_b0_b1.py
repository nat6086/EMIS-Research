import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from emis_research.mismatch.compare_b0_b1 import (MEASURES,bootstrap,metric_vector,
    sample_indices,strict_pair,summarize)


class ComparisonTests(unittest.TestCase):
    def frames(self):
        b0=pd.DataFrame(dict(audio_id=['a','b','c','d'],protocol='S1',outer_fold=[0,0,1,1],
            role='test',group_text=['g0','g0','g1','g1'],y_mismatch=[0,1,0,1],b0_score=[0,1,1,0]))
        b1=b0.drop(columns='b0_score').assign(b1_score=[.1,.9,.2,.8])
        return b0,b1

    def test_pairing_strict_and_order_invariant(self):
        b0,b1=self.frames()
        p=strict_pair(b0,b1)
        pd.testing.assert_frame_equal(p,strict_pair(b0.iloc[::-1],b1.sample(frac=1,random_state=42)))
        for bad in [b1.iloc[:-1],pd.concat([b1,b1.iloc[:1]]),b1.assign(y_mismatch=0),
                    b1.assign(group_text='bad'),b1.assign(outer_fold=2),b1.assign(role='train_oof')]:
            with self.assertRaises(ValueError): strict_pair(b0,bad)

    def test_whole_groups_multiplicity(self):
        idx=sample_indices([np.array([0,1]),np.array([2,3,4])],[1,1,0])
        np.testing.assert_array_equal(idx,[2,3,4,2,3,4,0,1])

    def test_seed_and_input_order_reproducible(self):
        a,b=self.frames();p=strict_pair(a,b)
        o,r,g=bootstrap(p,n_boot=30,seed=42)
        o2,r2,g2=bootstrap(strict_pair(a.iloc[::-1],b.iloc[::-1]),n_boot=30,seed=42)
        np.testing.assert_array_equal(o,o2);pd.testing.assert_frame_equal(r,r2);self.assertEqual(g,g2)
        self.assertFalse(r.equals(bootstrap(p,n_boot=30,seed=43)[1]))
        self.assertTrue(r.n_audio.eq(4).all())
        self.assertTrue(r.valid.all())

    def test_metrics_and_differences(self):
        a,b=self.frames();p=strict_pair(a,b)
        v=metric_vector(p.y_mismatch,p.b0_score,p.b1_score)
        self.assertEqual(v[0],roc_auc_score(p.y_mismatch,p.b0_score))
        self.assertEqual(v[4],average_precision_score(p.y_mismatch,p.b1_score))
        self.assertEqual(v[2],v[1]-v[0]);self.assertEqual(v[5],v[4]-v[3])
        p['b1_score']=p.b0_score
        _,r,_=bootstrap(p,20,42)
        self.assertTrue(r.delta_auroc.eq(0).all() and r.delta_ap.eq(0).all())

    def test_percentiles_and_invalid_exclusion(self):
        r=pd.DataFrame({m:[0.,1.,np.nan] for m in MEASURES}).assign(valid=[True,True,False])
        s=summarize(np.zeros(6),r)
        np.testing.assert_allclose(s.ci95_low,.025);np.testing.assert_allclose(s.ci95_high,.975)
        self.assertTrue(s.n_invalid.eq(1).all())
        self.assertFalse(s.loc[s.measure.str.startswith('delta'),'includes_zero'].any())

    def test_invalid_replicas_recorded_without_replacement(self):
        a,b=self.frames();a['y_mismatch']=[0,0,1,1];b['y_mismatch']=a.y_mismatch
        _,r,_=bootstrap(strict_pair(a,b),100,42)
        self.assertEqual(len(r),100)
        self.assertTrue((~r.valid).any())
        self.assertTrue(r.loc[~r.valid,MEASURES].isna().all().all())
        self.assertTrue(r.loc[~r.valid,'reason'].eq('single_ground_truth_class').all())


if __name__=='__main__': unittest.main()
