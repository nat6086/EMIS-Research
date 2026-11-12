"""M1-KRBF exploratori fix: reutilitza targets M1, mai ajusta SVM."""
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import roc_auc_score, average_precision_score

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from emis_research.mismatch.b0_labels import ROOT,LABELS,require,sha,dump_new
from emis_research.mismatch.b1_js import PA,PT,validate_probabilities
from emis_research.mismatch.m1_conditional import raw_prediction,residuals,fit_covariance,mahalanobis_squared,diagnostic
from emis_research.mismatch.compare_b0_b1 import bootstrap,summarize

VERSION='m1_krbf_centered_fixed_v1'
FILES=['scores.csv','reference_residuals.csv','models.json','metrics.csv',
       'bootstrap_summary.csv','bootstrap_replicas.csv','run_config.json','audit.json']


class CenteredKernelRidge:
    def fit(self,x,y):
        validate_probabilities(x);validate_probabilities(y)
        require(len(x)==len(y) and len(x)>=2,'targets invàlids')
        self.mean_=np.asarray(y).mean(axis=0)
        self.model_=KernelRidge(kernel='rbf',alpha=1.0,gamma=1.0).fit(x,np.asarray(y)-self.mean_)
        return self

    def predict(self,x):
        return self.model_.predict(x)+self.mean_

    def record(self):
        return dict(kernel='rbf',alpha=1.0,gamma=1.0,target_mean=self.mean_.tolist(),
                    dual_coef=self.model_.dual_coef_.tolist(),x_fit=self.model_.X_fit_.tolist())


def validate_targets(d,base,nested,plans):
    require(not base.duplicated(['outer_fold','audio_id']).any(),'targets base duplicats')
    require(not nested.duplicated(['outer_fold','scope','audio_id']).any(),'targets estrictes duplicats')
    group=d.set_index('audio_id').group_text.to_dict()
    for f in [base,nested]:
        require(f.role.eq('train_oof').all() and f.protocol.eq('S1').all(),'rol/protocol targets incorrecte')
        validate_probabilities(f[PA])
        meta=d.set_index('audio_id').loc[f.audio_id]
        for c in ['group_text','text_emotion','generation_emotion']:
            require(np.array_equal(meta[c],f[c]),'metadades targets diferents')
    require(len(plans)==5 and {p['outer_fold'] for p in plans}==set(range(5)),'plans incomplets')
    for p in plans:
        k=p['outer_fold'];tr=set(p['outer_train']);te=set(p['outer_test'])
        require(tr==set(d.loc[d.outer_fold!=k,'audio_id']) and te==set(d.loc[d.outer_fold==k,'audio_id']),'split extern diferent')
        require(not {group[x] for x in tr}&{group[x] for x in te},'fuga externa')
        b=base[base.outer_fold==k]
        require(set(b.audio_id)==tr,'cobertura base incorrecta')
        covered=[]
        for s in p['ridge_splits']:
            a=set(s['train']);v=set(s['validation']);covered.extend(v)
            require(a|v==tr and not {group[x] for x in a}&{group[x] for x in v},'fuga condicional')
            n=nested[(nested.outer_fold==k)&(nested.scope==f"residual_ridge{s['ridge_fold']}_target")]
            require(set(n.audio_id)==a,'cobertura targets estrictes incorrecta')
        require(len(covered)==len(tr) and set(covered)==tr,'cobertura residuals incorrecta')
        for t in p['acoustic_tasks']:
            a=set(t['final_svm_train']);v=set(t['predict'])
            require(not (a|v)&te and not {group[x] for x in a}&{group[x] for x in v},'fuga acústica')
            f=b if t['kind']=='base_oof' else nested[nested.outer_fold==k]
            require(set(f.loc[f.model_fit_id==t['task_id'],'audio_id'])==v,'procedència target incorrecta')
            if t['kind']=='ridge_target':
                held=set(next(s['validation'] for s in p['ridge_splits'] if s['ridge_fold']==t['ridge_fold']))
                require(not {group[x] for x in a|v}&{group[x] for x in held},'fuga de validació estricta')
            for c in t['calibration_splits']:
                ca=set(c['svm_train']);cv=set(c['calibrator_validation'])
                require(ca|cv==a and not {group[x] for x in ca}&{group[x] for x in cv},'fuga de calibratge')


def load_inputs():
    out=ROOT/'outputs/mismatch_v1'
    d=pd.read_csv(out/'m1_scores.csv').sort_values('audio_id').reset_index(drop=True)
    b1=pd.read_csv(out/'b1_scores.csv').set_index('audio_id')
    require(len(d)==1248 and d.audio_id.is_unique and d.role.eq('test').all() and d.protocol.eq('S1').all(),'test invàlid')
    require(b1.index.is_unique and len(b1)==len(d) and set(d.audio_id)==set(b1.index),'cobertura B1 incorrecta')
    b1=b1.loc[d.audio_id]
    for c in ['outer_fold','group_text','y_mismatch']:
        require(np.array_equal(d[c],b1[c]),'claus/etiquetes B1 diferents')
    np.testing.assert_allclose(d[PA+PT],b1[PA+PT],rtol=1e-14,atol=0)
    for cols in [PA,PT]:validate_probabilities(d[cols])
    man=pd.read_csv(ROOT/'data/processed/emis_manifest.csv',dtype=str).set_index('audio_id').loc[d.audio_id]
    splits=pd.read_csv(ROOT/'configs/splits/splits_v1.csv').set_index('audio_id').loc[d.audio_id]
    require(np.array_equal(man.text_category+'_'+man.text_id,d.group_text),'grups manifest diferents')
    require(np.array_equal((man.text_emotion!=man.generation_emotion).astype(int),d.y_mismatch),'ground truth diferent')
    require(np.array_equal(splits.fold_S1,d.outer_fold),'folds diferents')
    base=pd.read_csv(out/'m1_audio_train_oof.csv');nested=pd.read_csv(out/'m1_residual_targets.csv')
    plans=json.loads((out/'m1_partitions.json').read_text())
    cfg=json.loads((out/'m1_run_config.json').read_text())
    for n,h in cfg['original_hashes'].items():require(sha(out/n)==h,'artefacte previ alterat')
    for n,h in cfg['input_audit']['hashes'].items():
        if n=='manifest':require(sha(ROOT/'data/processed/emis_manifest.csv')==h,'manifest alterat')
        if n=='splits':require(sha(ROOT/'configs/splits/splits_v1.csv')==h,'splits alterats')
    require(len(base)==4992 and len(nested)==19968,'targets incomplets')
    validate_targets(d,base,nested,plans)
    return d,base,nested,plans


def run_fold(d,base,nested,plan):
    k=plan['outer_fold'];meta=d.set_index('audio_id');b=base[base.outer_fold==k].set_index('audio_id')
    refs=[];records=[]
    for s in plan['ridge_splits']:
        j=s['ridge_fold'];train=meta.loc[s['train']];val=meta.loc[s['validation']]
        train=train[train.y_mismatch==0];val=val[val.y_mismatch==0]
        targets=nested[(nested.outer_fold==k)&(nested.scope==f'residual_ridge{j}_target')].set_index('audio_id')
        model=CenteredKernelRidge().fit(train[PT].to_numpy(),targets.loc[train.index,PA].to_numpy())
        expected=raw_prediction(model,val[PT].to_numpy());r=residuals(b.loc[val.index,PA].to_numpy(),expected)
        table=val[['protocol','outer_fold','group_text','y_mismatch']].reset_index()
        table['role']='reference_residual';table['model_fit_id']=f'{VERSION}_fold{k}_ref{j}'
        for i,c in enumerate(LABELS):
            table[f'expected_{c}']=expected[:,i];table[f'residual_{c}']=r[:,i]
        refs.append(table)
        records.append(dict(model_fit_id=f'{VERSION}_fold{k}_ref{j}',train_audio_ids=train.index.tolist(),
            validation_audio_ids=val.index.tolist(),diagnostics=diagnostic(expected),**model.record()))
    reference=pd.concat(refs,ignore_index=True)
    train=meta.loc[plan['outer_train']];train=train[train.y_mismatch==0]
    require(reference.audio_id.is_unique and set(reference.audio_id)==set(train.index),'cobertura residuals')
    cov=fit_covariance(reference[[f'residual_{c}' for c in LABELS]].to_numpy())
    model=CenteredKernelRidge().fit(train[PT].to_numpy(),b.loc[train.index,PA].to_numpy())
    scored=meta.loc[plan['outer_test']].reset_index().copy()
    expected=raw_prediction(model,scored[PT].to_numpy());r=residuals(scored[PA].to_numpy(),expected)
    scored['m1_krbf_score']=mahalanobis_squared(r,cov)
    scored['m1_krbf_fit_id']=f'{VERSION}_fold{k}';scored['m1_krbf_version']=VERSION
    for i,c in enumerate(LABELS):
        scored[f'krbf_expected_{c}']=expected[:,i];scored[f'krbf_residual_{c}']=r[:,i]
    record=dict(outer_fold=k,references=records,final_model=model.record(),train_audio_ids=train.index.tolist(),
        covariance_audio_ids=reference.audio_id.tolist(),residual_mean=cov.location_.tolist(),
        covariance=cov.covariance_.tolist(),shrinkage=float(cov.shrinkage_),test_diagnostics=diagnostic(expected))
    return scored,reference,record


def metrics(d):
    rows=[]
    for k,p in [(str(k),v) for k,v in d.groupby('outer_fold')]+[('global',d)]:
        for name,col in [('B1','b1_score'),('M1','m1_score'),('M1-KRBF','m1_krbf_score')]:
            valid=p.y_mismatch.nunique()==2
            rows.append(dict(protocol='S1',outer_fold=k,method=name,n=len(p),
                auroc=float(roc_auc_score(p.y_mismatch,p[col])) if valid else None,
                average_precision=float(average_precision_score(p.y_mismatch,p[col])) if valid else None,
                status='ok' if valid else 'single_class'))
    return pd.DataFrame(rows)


def comparisons(d,n_boot=10000):
    summaries=[];replicas=[]
    for name,col in [('M1','m1_score'),('B1','b1_score')]:
        p=d[['group_text','y_mismatch',col,'m1_krbf_score']].rename(columns={col:'b0_score','m1_krbf_score':'b1_score'})
        obs,r,_=bootstrap(p,n_boot=n_boot,seed=42,progress=n_boot==10000)
        s=summarize(obs,r)
        mapping={'auroc_b0':f'auroc_{name}','auroc_b1':'auroc_KRBF','ap_b0':f'ap_{name}','ap_b1':'ap_KRBF'}
        s['measure']=s.measure.map(lambda x:mapping.get(x,x));s['comparison']=f'KRBF_minus_{name}'
        summaries.append(s);replicas.append(r.rename(columns=mapping).assign(comparison=f'KRBF_minus_{name}'))
    return pd.concat(summaries,ignore_index=True),pd.concat(replicas,ignore_index=True)


def main():
    out=ROOT/'outputs/mismatch_v1'
    require(not any((out/('m1_krbf_'+f)).exists() for f in FILES),'sortides KRBF existents')
    originals={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    d,base,nested,plans=load_inputs()
    config=dict(version=VERSION,kernel='rbf',alpha=1.0,gamma=1.0,center_targets=True,
        kernel_ridge_fits=30,covariance_fits=5,new_svm_fits=0,bootstrap_replicas=10000,seed=42,
        original_hashes=originals,script_sha256=sha(__file__),python=platform.python_version(),
        numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
        limitation='Exploratory post-S1, fixed non-optimized hyperparameters; not independent validation. OOF excludes fitting/calibration, not prior SVM hyperparameter selection.')
    dump_new(out/'m1_krbf_run_config.json',config)
    scored=[];refs=[];models=[]
    for plan in plans:
        print(f"KRBF fold {plan['outer_fold']}: 6 ajustos, cap SVM",flush=True)
        s,r,m=run_fold(d,base,nested,plan);scored.append(s);refs.append(r);models.append(m)
    scored=pd.concat(scored,ignore_index=True).sort_values('audio_id').reset_index(drop=True)
    require(scored.audio_id.is_unique and len(scored)==1248,'cobertura KRBF incorrecta')
    np.testing.assert_array_equal(scored[PA+PT],d[PA+PT])
    comparison=metrics(scored);print(comparison.to_string(index=False),flush=True)
    summary,reps=comparisons(scored)
    for n,h in originals.items():require(sha(out/n)==h,'artefacte previ alterat')
    for name,t in [('scores.csv',scored),('reference_residuals.csv',pd.concat(refs,ignore_index=True)),
                   ('metrics.csv',comparison),('bootstrap_summary.csv',summary),('bootstrap_replicas.csv',reps)]:
        t.to_csv(out/('m1_krbf_'+name),index=False,mode='x')
    dump_new(out/'m1_krbf_models.json',models)
    dump_new(out/'m1_krbf_audit.json',dict(n_test=len(scored),n_groups=scored.group_text.nunique(),
        kernel_ridge_fits=30,covariance_fits=5,new_svm_fits=0,prior_artifacts='hashes_unchanged',
        test_vectors='exactly_equal_M1_in_memory',valid_replicas=reps.groupby('comparison').valid.sum().to_dict(),
        test_outside_simplex=sum(m['test_diagnostics']['outside_simplex'] for m in models),
        reference_outside_simplex=sum(r['diagnostics']['outside_simplex'] for m in models for r in m['references'])))
    print(summary.to_string(index=False))


if __name__=='__main__':main()
