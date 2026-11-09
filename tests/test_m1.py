import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from emis_research.mismatch.m1_conditional import (ALPHA,LABELS,PA,PT,count_fits,
    diagnostic,fit_acoustic,fit_covariance,load_resources,mahalanobis_squared,
    paired_bootstrap,plan_fold,raw_prediction,residuals,ridge_fit,run_fold,serialize_plan)


def fixture():
    rng=np.random.default_rng(42)
    d=pd.DataFrame(dict(audio_id=[f'a{i:03}' for i in range(80)],
        group_text=np.repeat([f'g{i:02}' for i in range(20)],4),
        generation_emotion=LABELS*20,text_emotion=np.repeat([LABELS[i%4] for i in range(20)],4),
        fold_S1=np.repeat([0,1],[16,64])))
    x=rng.normal(size=(80,1,3))
    pt=np.repeat(rng.dirichlet(np.ones(4),20),4,axis=0)
    plan=plan_fold(d,0)
    t=d[d.fold_S1==0][['audio_id','group_text','fold_S1']].copy()
    for c in PA:t[c]=.25
    for i,c in enumerate(PT):t[c]=pt[:16,i]
    t['protocol']='S1';t['outer_fold']=0;t['role']='test'
    t['y_mismatch']=(d.iloc[:16].text_emotion!=d.iloc[:16].generation_emotion).astype(int).to_numpy()
    t['b1_score']=np.linspace(0,1,len(t))
    return d,x,pt,t,plan,dict(layer=0,params={'C':10,'gamma':.001})


def fake_acoustic(d,x,task,selection,n_jobs):
    # Targets sintètics deterministes, dependents només del train de la tasca
    # i de les features de la mostra predita. Cap entrenament sobre EMIS.
    mean=x[task['train'],0].mean(axis=0)
    z=x[task['predict'],0]+mean
    z=np.column_stack([z,np.zeros(len(z))]);z-=z.max(axis=1,keepdims=True)
    e=np.exp(z)
    return e/e.sum(axis=1,keepdims=True)


class M1Tests(unittest.TestCase):
    def test_real_audit_and_total_count_without_training(self):
        d,_,_,t,_,plans,_=load_resources()
        self.assertEqual(len(d),1248);self.assertEqual(len(t),1248)
        self.assertEqual(count_fits(plans),dict(base_oof_svm=150,strict_residual_svm=750,
            total_svm=900,calibrated_chains=150,ridge_fits=30))

    def test_strict_partitions_exclude_ridge_validation_everywhere(self):
        d,_,_,_,p,_=fixture();external=set(p['test'])
        for task in p['tasks']:
            a,v=task['train'],task['predict']
            self.assertFalse(set(a)&external);self.assertFalse(set(v)&external)
            self.assertFalse(set(d.iloc[a].group_text)&set(d.iloc[v].group_text))
            if task['kind']=='ridge_target':
                held=set(p['ridge_splits'][task['ridge_fold']][1])
                self.assertFalse((set(a)|set(v))&held)
                for aa,vv in task['calibration']:
                    self.assertFalse((set(aa)|set(vv))&held)
            for aa,vv in task['calibration']:
                self.assertEqual(set(aa)|set(vv),set(a))
                self.assertFalse(set(d.iloc[aa].group_text)&set(d.iloc[vv].group_text))
        self.assertEqual(len(serialize_plan(d,p)['acoustic_tasks']),30)
        self.assertEqual(serialize_plan(d,p),serialize_plan(d,plan_fold(d,0)))

    def test_actual_small_calibration_ignores_heldout(self):
        d,x,_,_,p,s=fixture();task=p['tasks'][0]
        first=fit_acoustic(d,x,task,s)
        altered=d.copy();altered.loc[task['predict'],'generation_emotion']='sad'
        # Perturb labels of prediction samples: no effect on calibration or predictions.
        np.testing.assert_array_equal(first,fit_acoustic(altered,x,task,s))
        np.testing.assert_allclose(first.sum(axis=1),1)

    def test_ridge_residuals_covariance_and_manual_mahalanobis(self):
        rng=np.random.default_rng(42)
        pt=rng.dirichlet(np.ones(4),30);pa=rng.dirichlet(np.ones(4),30)
        ridge=ridge_fit(pt,pa);self.assertEqual(ridge.alpha,1.0);self.assertTrue(ridge.fit_intercept)
        expected=raw_prediction(ridge,pt);r=residuals(pa,expected)
        np.testing.assert_array_equal(r,pa-expected)
        cov=fit_covariance(r);self.assertEqual(cov.covariance_.shape,(3,3))
        self.assertTrue((np.linalg.eigvalsh(cov.covariance_)>0).all())
        scores=mahalanobis_squared(r,cov)
        manual=np.array([(v[:3]-cov.location_)@np.linalg.inv(cov.covariance_)@(v[:3]-cov.location_) for v in r])
        np.testing.assert_allclose(scores,manual)
        self.assertTrue(np.isfinite(scores).all() and (scores>=0).all())
        with self.assertRaises(ValueError):fit_covariance(np.zeros((10,4)))

    def test_raw_outputs_are_not_clipped(self):
        class Raw:
            def predict(self,x):return np.tile([1.2,-.2,.1,-.1],(len(x),1))
        p=raw_prediction(Raw(),np.ones((2,4))*.25)
        self.assertEqual(p[0,0],1.2);self.assertEqual(diagnostic(p)['outside_simplex'],2)

    def test_full_synthetic_fold_and_external_test_mutation(self):
        d,x,pt,t,p,s=fixture()
        first=run_fold(d,x,pt,t,s,p,acoustic_fitter=fake_acoustic)
        mutated=d.copy();mutated.loc[p['test'],'generation_emotion']='sad'
        x2=x.copy();x2[p['test']]=10000
        pt2=pt.copy();pt2[p['test']]=.25
        second=run_fold(mutated,x2,pt2,t,s,p,acoustic_fitter=fake_acoustic)
        self.assertEqual(first[4],second[4])
        scored,oof,targets,refs,record=first
        np.testing.assert_array_equal(scored[PA+PT],t[PA+PT])
        self.assertTrue(scored.m1_score.ge(0).all())
        self.assertTrue(refs.text_emotion.eq(refs.generation_emotion).all())
        congruent=set(d.loc[(d.fold_S1!=0)&(d.text_emotion==d.generation_emotion),'audio_id'])
        self.assertEqual(set(record['ridge_train_audio_ids']),congruent)
        self.assertEqual(set(record['covariance_audio_ids']),congruent)
        self.assertEqual(len(oof),64);self.assertEqual(len(targets),256)
        # Mutating a Ridge validation group's features cannot alter that Ridge's parameters.
        held=p['ridge_splits'][0][1];xx=x.copy();xx[held]=1000
        changed=run_fold(d,xx,pt,t,s,p,acoustic_fitter=fake_acoustic)
        a,b=record['reference_ridges'][0],changed[4]['reference_ridges'][0]
        self.assertEqual(a['coef'],b['coef']);self.assertEqual(a['intercept'],b['intercept'])
        changed_test=t.copy();changed_test[PA]=np.tile([.7,.1,.1,.1],(len(t),1))
        altered_test=run_fold(d,x,pt,changed_test,s,p,acoustic_fitter=fake_acoustic)[4]
        self.assertEqual(record['final_ridge'],altered_test['final_ridge'])
        self.assertEqual(record['covariance'],altered_test['covariance'])
        self.assertEqual(record['residual_mean'],altered_test['residual_mean'])
        with self.assertRaises(ValueError):
            run_fold(d,x,pt,t.iloc[:-1],s,p,acoustic_fitter=fake_acoustic)
        summary,reps=paired_bootstrap(scored,n_boot=20)
        self.assertEqual(len(reps),20);self.assertIn('auroc_m1',reps)
        self.assertEqual(set(summary.comparison),{'M1_minus_B1'})


if __name__=='__main__':unittest.main()
