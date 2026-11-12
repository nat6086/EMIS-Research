import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from emis_research.mismatch.m1_kernel_rbf import (CenteredKernelRidge,PA,PT,
    load_inputs,run_fold,comparisons,validate_targets)


class KernelTests(unittest.TestCase):
    def test_centering_fixed_parameters_and_manual_prediction(self):
        rng=np.random.default_rng(42);x=rng.dirichlet(np.ones(4),20);y=rng.dirichlet(np.ones(4),20)
        model=CenteredKernelRidge().fit(x,y)
        np.testing.assert_array_equal(model.mean_,y.mean(axis=0))
        self.assertEqual(model.model_.alpha,1);self.assertEqual(model.model_.gamma,1)
        pred=model.predict(x[:3]);np.testing.assert_allclose(pred.sum(axis=1),1,atol=1e-14)
        dual=np.linalg.solve(rbf_kernel(x,gamma=1)+np.eye(20),y-y.mean(axis=0))
        np.testing.assert_allclose(pred,rbf_kernel(x[:3],x,gamma=1)@dual+y.mean(axis=0))

    def test_no_clipping(self):
        model=CenteredKernelRidge()
        class Raw:
            def predict(self,x):return np.tile([1.,-1.,0.,0.],(len(x),1))
        model.mean_=np.full(4,.25);model.model_=Raw()
        np.testing.assert_array_equal(model.predict(np.zeros((1,4))),[[1.25,-.75,.25,.25]])

    def test_real_targets_coverage_and_leakage_rejection_no_svm(self):
        with patch('sklearn.svm.SVC.fit',side_effect=AssertionError('SVM prohibit')):
            d,b,n,plans=load_inputs()
        self.assertEqual(len(d),1248);self.assertEqual(len(plans),5)
        with self.assertRaises(ValueError):validate_targets(d,b.iloc[:-1],n,plans)
        altered=n.copy();altered.loc[0,'group_text']='wrong'
        with self.assertRaises(ValueError):validate_targets(d,b,altered,plans)

    def fixture(self):
        rng=np.random.default_rng(42);ids=[f'a{i:02}' for i in range(40)]
        x=np.repeat(rng.dirichlet(np.ones(4),10),4,axis=0);y=rng.dirichlet(np.ones(4),40)
        d=pd.DataFrame(dict(audio_id=ids,group_text=np.repeat([f'g{i}' for i in range(10)],4),
            y_mismatch=np.tile([0,1,1,1],10),outer_fold=np.repeat([1,0],[32,8]),
            protocol='S1',role='test',b1_score=np.arange(40)/40,m1_score=np.arange(40)/10))
        for i,c in enumerate(PA):d[c]=y[:,i]
        for i,c in enumerate(PT):d[c]=x[:,i]
        base=d.iloc[:32][['audio_id']+PA].assign(outer_fold=0)
        splits=[];targets=[]
        for j,gg in enumerate(np.array_split(np.arange(8),5)):
            v=[ids[i] for g in gg for i in range(g*4,g*4+4)];a=sorted(set(ids[:32])-set(v))
            splits.append(dict(ridge_fold=j,train=a,validation=v))
            targets.append(base[base.audio_id.isin(a)].assign(scope=f'residual_ridge{j}_target'))
        return d,base,pd.concat(targets),dict(outer_fold=0,outer_train=ids[:32],outer_test=ids[32:],ridge_splits=splits)

    def test_synthetic_fold_training_masks_and_test_independence(self):
        d,b,n,p=self.fixture()
        with patch('sklearn.svm.SVC.fit',side_effect=AssertionError('SVM prohibit')):
            score,ref,record=run_fold(d,b,n,p)
        self.assertEqual(len(score),8);self.assertEqual(len(ref),8)
        self.assertTrue(ref.y_mismatch.eq(0).all())
        self.assertEqual(record['train_audio_ids'],[f'a{i:02}' for i in range(0,32,4)])
        changed=d.copy();changed.loc[32:,PT]=.25;changed.loc[32:,PA]=.25
        _,_,record2=run_fold(changed,b,n,p)
        self.assertEqual(record['final_model'],record2['final_model'])
        self.assertEqual(record['covariance'],record2['covariance'])
        np.testing.assert_array_equal(score[PA+PT],d.iloc[32:][PA+PT])
        self.assertTrue(score.m1_krbf_score.ge(0).all())
        s,r=comparisons(score,n_boot=10);s2,r2=comparisons(score,n_boot=10)
        pd.testing.assert_frame_equal(s,s2);pd.testing.assert_frame_equal(r,r2)
        self.assertEqual(set(s.comparison),{'KRBF_minus_M1','KRBF_minus_B1'})


if __name__=='__main__':unittest.main()
