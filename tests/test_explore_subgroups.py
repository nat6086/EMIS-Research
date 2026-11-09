import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from emis_research.mismatch.explore_subgroups import (ROOT,align,examples,relative_scores,
    subgroup_metrics,subgroup_bootstrap,within_text)


class ExploratoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=ROOT/'outputs/mismatch_v1'
        cls.frames=[pd.read_csv(p/n,dtype={'reference_id':str,'text_id':str}) for n in
            ['mismatch_scores.csv','b1_scores.csv','m1_scores.csv']]
        cls.man=pd.read_csv(ROOT/'data/processed/emis_manifest.csv',dtype=str)
        cls.splits=pd.read_csv(ROOT/'configs/splits/splits_v1.csv')
        cls.data=align(*cls.frames,cls.man,cls.splits)

    def test_strict_correspondence_and_groups(self):
        d=self.data
        self.assertEqual(len(d),1248);self.assertEqual(d.group_text.nunique(),104)
        self.assertEqual(set(d.text_cue_type),{'explicit','implicit','neutral'})
        pd.testing.assert_frame_equal(d,align(*[f.iloc[::-1] for f in self.frames],self.man,self.splits))
        for bad in [self.frames[2].iloc[:-1],pd.concat([self.frames[2],self.frames[2].iloc[:1]]),
                    self.frames[2].assign(y_mismatch=1),self.frames[2].assign(outer_fold=8)]:
            with self.assertRaises(ValueError):align(*self.frames[:2],bad,self.man,self.splits)

    def test_subgroup_metrics_known_case(self):
        d=pd.DataFrame(dict(text_cue_type=['x']*4,group_text=['a','a','b','b'],
            y_mismatch=[0,1,0,1],b0_score=[0,1,1,0],b1_score=[.1,.9,.2,.8],m1_score=[.9,.1,.8,.2]))
        r=subgroup_metrics(d).iloc[0]
        self.assertEqual(r.auroc_B0,.5);self.assertEqual(r.auroc_B1,1);self.assertEqual(r.auroc_M1,0)
        self.assertEqual(r.auroc_M1_minus_B1,-1);self.assertEqual(r.auroc_B1_minus_B0,.5)
        self.assertEqual(r.ap_B1,1)

    def test_seed_reproducibility(self):
        d=self.data[self.data.text_cue_type=='neutral']
        a,r=subgroup_bootstrap(d,n_boot=10);b,s=subgroup_bootstrap(d,n_boot=10)
        pd.testing.assert_frame_equal(a,b);pd.testing.assert_frame_equal(r,s)
        self.assertTrue(r.n_audio.eq(len(d)).all())
        self.assertTrue(r.valid.all())

    def test_relative_rank_invariance_and_examples(self):
        d=self.data;a=relative_scores(d);b=relative_scores(d.assign(m1_score=3*d.m1_score+7))
        np.testing.assert_array_equal(a.percentile_M1,b.percentile_M1)
        e=examples(a);self.assertEqual(len(e),9)
        self.assertTrue(e.groupby('example_category').group_text.nunique().eq(3).all())

    def test_within_text_pairs_and_ties(self):
        p,s=within_text(self.data)
        self.assertEqual(len(p),936)
        self.assertTrue(p.groupby('group_text').size().eq(9).all())
        self.assertTrue(s.filter(like='concordance_').ge(0).all().all())
        self.assertTrue(s.filter(like='concordance_').le(1).all().all())
        flat=self.data.assign(b0_score=0,b1_score=0,m1_score=0)
        _,same=within_text(flat)
        self.assertTrue(same.filter(like='concordance_').eq(.5).all().all())
        with self.assertRaises(ValueError):within_text(self.data.iloc[:-1])


if __name__=='__main__':unittest.main()
