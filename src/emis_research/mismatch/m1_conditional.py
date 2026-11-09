"""M1 S1 estricte. Per defecte audita i compta; --train requereix autorització."""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emis_research.mismatch.b0_labels import ROOT, LABELS, SEED, dump_new, load_audit, make_model, require, sha
from emis_research.mismatch.b1_js import PA, PT, canonical_probabilities, js_divergence, load_b0, validate_probabilities
from emis_research.mismatch.compare_b0_b1 import bootstrap, summarize

VERSION = 'm1_s1_strict_alpha1_v1'
ALPHA = 1.0
FILES = ['m1_run_config.json', 'm1_audio_train_oof.csv', 'm1_residual_targets.csv',
         'm1_reference_residuals.csv', 'm1_scores.csv', 'm1_metrics.csv',
         'm1_models.json', 'm1_partitions.json', 'm1_audit_report.json',
         'm1_bootstrap_summary.csv', 'm1_bootstrap_replicas.csv']
LIMITATION = ('OOF independent of estimator fitting and calibration, but not of prior '
              'hyperparameter selection. Fixed B1 test vectors. Conditional bootstrap '
              'of fixed scores excludes training variability. Synthetic generation mismatch.')


def grouped_splits(d, indices):
    indices = np.asarray(indices, dtype=int)
    part = d.iloc[indices]
    require(len(np.unique(indices)) == len(indices) and part.group_text.nunique() >= 5,
            'cinc folds agrupats inviables')
    splits = []
    for a,b in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(
            np.zeros(len(part)), part.generation_emotion, part.group_text):
        train, val = indices[a], indices[b]
        require(not set(d.iloc[train].group_text) & set(d.iloc[val].group_text), 'fuga de grups')
        for rows in [train,val]:
            require(set(d.iloc[rows].generation_emotion) == set(LABELS), 'classes incompletes')
        splits.append((train,val))
    require(np.array_equal(np.sort(np.concatenate([v for _,v in splits])), np.sort(indices)), 'cobertura OOF invàlida')
    return splits


def plan_fold(d, k):
    tr = np.flatnonzero(d.fold_S1.to_numpy() != k)
    te = np.flatnonzero(d.fold_S1.to_numpy() == k)
    require(d.audio_id.is_unique and len(te) > 0, 'fold extern invàlid')
    require(not set(d.iloc[tr].group_text) & set(d.iloc[te].group_text), 'fuga externa')
    top = grouped_splits(d,tr)
    tasks = []
    for j,(a,v) in enumerate(top):
        tasks.append(dict(task_id=f'outer{k}_base{j}',kind='base_oof',ridge_fold=j,
                          train=a,predict=v,calibration=grouped_splits(d,a)))
        for h,(aa,vv) in enumerate(grouped_splits(d,a)):
            tasks.append(dict(task_id=f'outer{k}_ridge{j}_target{h}',kind='ridge_target',
                              ridge_fold=j,train=aa,predict=vv,calibration=grouped_splits(d,aa)))
    return dict(outer_fold=k,train=tr,test=te,ridge_splits=top,tasks=tasks)


def count_fits(plans):
    base = sum(len(t['calibration'])+1 for p in plans for t in p['tasks'] if t['kind']=='base_oof')
    strict = sum(len(t['calibration'])+1 for p in plans for t in p['tasks'] if t['kind']=='ridge_target')
    return dict(base_oof_svm=base,strict_residual_svm=strict,total_svm=base+strict,
                calibrated_chains=sum(len(p['tasks']) for p in plans),ridge_fits=6*len(plans))


def fit_acoustic(d,cube,task,selection,n_jobs=1):
    tr, pred = task['train'], task['predict']
    require(not set(d.iloc[tr].group_text) & set(d.iloc[pred].group_text), 'fuga acústica')
    local = {int(ix):i for i,ix in enumerate(tr)}
    cv = []
    for a,b in task['calibration']:
        require(set(a)|set(b) == set(tr) and not set(a)&set(b), 'partició de calibratge invàlida')
        require(not set(d.iloc[a].group_text) & set(d.iloc[b].group_text), 'fuga calibratge')
        cv.append((np.array([local[int(i)] for i in a]),np.array([local[int(i)] for i in b])))
    model = CalibratedClassifierCV(make_model('SVM-RBF',selection['params']),
                method='sigmoid',ensemble=False,cv=cv,n_jobs=n_jobs)
    layer = selection['layer']
    model.fit(cube[tr,layer],d.iloc[tr].generation_emotion.to_numpy())
    p = canonical_probabilities(model.predict_proba(cube[pred,layer]),model.classes_)
    return p


def ridge_fit(pt,pa):
    validate_probabilities(pt);validate_probabilities(pa)
    require(len(pt)==len(pa) and len(pt)>=2, 'targets Ridge invàlids')
    return Ridge(alpha=ALPHA,fit_intercept=True).fit(pt,pa)


def raw_prediction(model,pt):
    p = model.predict(pt)
    require(p.shape == (len(pt),4) and np.isfinite(p).all(), 'sortida Ridge invàlida')
    require(np.allclose(p.sum(axis=1),1,atol=1e-6,rtol=0), 'Ridge no conserva suma 1')
    return p


def residuals(pa,expected):
    require(pa.shape == expected.shape and pa.ndim==2 and pa.shape[1]==4, 'shape residual invàlid')
    r = pa-expected
    require(np.isfinite(r).all() and np.allclose(r.sum(axis=1),0,atol=1e-6,rtol=0), 'residual invàlid')
    return r


def fit_covariance(r):
    require(r.ndim==2 and r.shape[1]==4 and len(r)>=2 and np.isfinite(r).all(), 'residuals de referència invàlids')
    cov = LedoitWolf().fit(r[:,:3])
    require(np.isfinite(cov.covariance_).all() and np.linalg.eigvalsh(cov.covariance_).min()>0,
            'covariància no definida positiva; sense correcció silenciosa')
    return cov


def mahalanobis_squared(r,cov):
    centered = r[:,:3]-cov.location_
    solved = np.linalg.solve(cov.covariance_,centered.T).T
    score = np.einsum('ij,ij->i',centered,solved)
    require(np.isfinite(score).all() and (score>=0).all(), 'Mahalanobis invàlid')
    return score


def ridge_record(model):
    return dict(alpha=ALPHA,fit_intercept=True,coef=model.coef_.tolist(),intercept=model.intercept_.tolist())


def diagnostic(p):
    return dict(n=len(p),outside_simplex=int(((p<0)|(p>1)).any(axis=1).sum()),
                min=float(p.min()),max=float(p.max()),max_sum_error=float(np.abs(p.sum(axis=1)-1).max()))


def probability_rows(d,indices,p,fold,task_id,scope):
    rows = d.iloc[indices][['audio_id','group_text','text_emotion','generation_emotion']].copy()
    for i,c in enumerate(PA): rows[c]=p[:,i]
    return rows.assign(protocol='S1',outer_fold=fold,role='train_oof',model_fit_id=task_id,
                       scope=scope,experiment_version=VERSION)


def run_fold(d,cube,pt,test,selection,plan,n_jobs=1,acoustic_fitter=fit_acoustic):
    k,tr = plan['outer_fold'],plan['train']
    base = np.full((len(d),4),np.nan)
    target_tables,base_tables,refs,models = [],[],[],[]
    congruent = (d.text_emotion==d.generation_emotion).to_numpy()
    tasks = plan['tasks']
    for task in [t for t in tasks if t['kind']=='base_oof']:
        p=acoustic_fitter(d,cube,task,selection,n_jobs)
        base[task['predict']]=p
        base_tables.append(probability_rows(d,task['predict'],p,k,task['task_id'],'final_ridge_target'))
    validate_probabilities(base[tr])
    for j,(a,v) in enumerate(plan['ridge_splits']):
        nested=np.full((len(d),4),np.nan)
        for task in [t for t in tasks if t['kind']=='ridge_target' and t['ridge_fold']==j]:
            p=acoustic_fitter(d,cube,task,selection,n_jobs)
            nested[task['predict']]=p
            target_tables.append(probability_rows(d,task['predict'],p,k,task['task_id'],f'residual_ridge{j}_target'))
        validate_probabilities(nested[a])
        train_c=a[congruent[a]];val_c=v[congruent[v]]
        ridge=ridge_fit(pt[train_c],nested[train_c])
        expected=raw_prediction(ridge,pt[val_c]);r=residuals(base[val_c],expected)
        rows=probability_rows(d,val_c,base[val_c],k,f'outer{k}_base{j}','reference_residual')
        rows['ridge_fit_id']=f'outer{k}_ridge{j}'
        for i,c in enumerate(LABELS):
            rows[f'expected_{c}']=expected[:,i];rows[f'residual_{c}']=r[:,i]
        refs.append(rows)
        models.append(dict(ridge_fit_id=f'outer{k}_ridge{j}',train_audio_ids=d.iloc[train_c].audio_id.tolist(),
                           validation_audio_ids=d.iloc[val_c].audio_id.tolist(),prediction_diagnostics=diagnostic(expected),**ridge_record(ridge)))
    reference=pd.concat(refs,ignore_index=True)
    require(reference.audio_id.is_unique and set(reference.audio_id)==set(d.iloc[tr[congruent[tr]]].audio_id),
            'cobertura residuals congruents invàlida')
    cov=fit_covariance(reference[[f'residual_{c}' for c in LABELS]].to_numpy())
    tc=tr[congruent[tr]];ridge=ridge_fit(pt[tc],base[tc])
    ids=d.iloc[plan['test']].audio_id
    require(test.audio_id.is_unique and set(test.audio_id)==set(ids),'test B1 incompatible')
    scored=test.set_index('audio_id').loc[ids].reset_index().copy()
    expected=raw_prediction(ridge,scored[PT].to_numpy());r=residuals(scored[PA].to_numpy(),expected)
    scored['m1_score']=mahalanobis_squared(r,cov)
    for i,c in enumerate(LABELS):
        scored[f'm1_expected_{c}']=expected[:,i];scored[f'm1_residual_{c}']=r[:,i]
    scored['m1_fit_id']=f'{VERSION}_fold{k}';scored['m1_experiment_version']=VERSION
    record=dict(outer_fold=k,reference_ridges=models,final_ridge=ridge_record(ridge),
                ridge_train_audio_ids=d.iloc[tc].audio_id.tolist(),covariance_audio_ids=reference.audio_id.tolist(),
                residual_mean=cov.location_.tolist(),covariance=cov.covariance_.tolist(),shrinkage=float(cov.shrinkage_),
                coordinates=LABELS[:3],score='squared_mahalanobis_centered',
                test_prediction_diagnostics=diagnostic(expected),
                acoustic_oof_mean=base[tr].mean(axis=0).tolist(),
                acoustic_test_mean=scored[PA].mean().tolist())
    return scored,pd.concat(base_tables),pd.concat(target_tables),reference,record


def serialize_plan(d,plan):
    ids=lambda ix:d.iloc[ix].audio_id.tolist()
    return dict(outer_fold=plan['outer_fold'],outer_train=ids(plan['train']),outer_test=ids(plan['test']),
        ridge_splits=[dict(ridge_fold=j,train=ids(a),validation=ids(v)) for j,(a,v) in enumerate(plan['ridge_splits'])],
        acoustic_tasks=[dict(task_id=t['task_id'],kind=t['kind'],ridge_fold=t['ridge_fold'],
            final_svm_train=ids(t['train']),predict=ids(t['predict']),
            calibration_splits=[dict(svm_train=ids(a),calibrator_validation=ids(b)) for a,b in t['calibration']])
            for t in plan['tasks']])


def load_resources():
    d,cube,q,audit=load_audit()
    _,selections,_=load_b0(d,audit)
    out=ROOT/'outputs/mismatch_v1'
    config=json.loads((out/'b1_run_config.json').read_text())
    require(config['input_hashes']==audit['hashes'] and config['ensemble'] is False and config['method']=='sigmoid',
            'configuració B1 incompatible')
    for name,h in config['b0_hashes'].items(): require(sha(out/name)==h,'B0 alterat')
    test=pd.read_csv(out/'b1_scores.csv')
    require(test.audio_id.is_unique and set(test.audio_id)==set(d.audio_id) and test.role.eq('test').all()
            and test.protocol.eq('S1').all(),'cobertura B1 incorrecta')
    check=test.set_index('audio_id').loc[d.audio_id]
    require(np.array_equal(check.outer_fold,d.fold_S1) and np.array_equal(check.group_text,d.group_text),'alineació B1 incorrecta')
    require(np.array_equal(check.y_mismatch,(d.text_emotion!=d.generation_emotion).astype(int)),'ground truth B1 incorrecte')
    validate_probabilities(check[PA]);validate_probabilities(check[PT])
    np.testing.assert_allclose(js_divergence(check[PA],check[PT]),check.b1_score,atol=1e-14,rtol=0)
    pt=q.set_index('text').loc[d.text,[f'p_{c}' for c in LABELS]].to_numpy()
    # El CSV B1 ha serialitzat de nou Qwen; admet només arrodoniment de lectura.
    np.testing.assert_allclose(pt,check[PT].to_numpy(),rtol=1e-14,atol=0)
    pt=check[PT].to_numpy().copy()
    for r in selections:
        require(r['layer']==3 and r['params']=={'C':10,'gamma':.001},'selecció acústica no validada')
    plans=[plan_fold(d,k) for k in range(5)]
    return d,cube,pt,test,selections,plans,audit


def metrics(scored):
    rows=[]
    for k,part in [(str(k),v) for k,v in scored.groupby('outer_fold')]+[('global',scored)]:
        for method,col in [('B1','b1_score'),('M1','m1_score')]:
            y=part.y_mismatch;valid=y.nunique()==2
            rows.append(dict(protocol='S1',outer_fold=k,method=method,n=len(part),
                 auroc=float(roc_auc_score(y,part[col])) if valid else None,
                 average_precision=float(average_precision_score(y,part[col])) if valid else None,
                 status='ok' if valid else 'single_ground_truth_class'))
    return pd.DataFrame(rows)


def paired_bootstrap(scored,n_boot=10000):
    # Reutilitza el motor aparellat existent: noms interns B0/B1 reanomenats a la sortida.
    paired=scored[['group_text','y_mismatch','b1_score','m1_score']].rename(columns={'b1_score':'b0_score','m1_score':'b1_score'})
    observed,reps,_=bootstrap(paired,n_boot=n_boot,seed=SEED,progress=n_boot==10000)
    summary=summarize(observed,reps)
    mapping={'auroc_b0':'auroc_b1','auroc_b1':'auroc_m1','ap_b0':'ap_b1','ap_b1':'ap_m1'}
    summary['measure']=summary.measure.map(lambda x:mapping.get(x,x))
    summary['comparison']='M1_minus_B1'
    return summary,reps.rename(columns=mapping)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',action='store_true');ap.add_argument('--n-jobs',type=int,default=1)
    args=ap.parse_args();out=ROOT/'outputs/mismatch_v1'
    require(not any((out/f).exists() for f in FILES),'sortides M1 existents; no se sobreescriuen')
    d,cube,pt,test,selections,plans,audit=load_resources()
    counts=count_fits(plans);print(json.dumps(counts),flush=True)
    if not args.train:
        print('Auditoria i particions OK; cap entrenament EMIS executat. Cal autorització per --train.')
        return
    original_hashes={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    source=[Path(__file__),ROOT/'src/emis_research/mismatch/b0_labels.py',ROOT/'src/emis_research/audio_wavlm_predict.py',
            ROOT/'src/emis_research/mismatch/b1_js.py',ROOT/'src/emis_research/mismatch/compare_b0_b1.py']
    dump_new(out/'m1_run_config.json',dict(version=VERSION,alpha=ALPHA,fit_intercept=True,
        seed=SEED,folds=5,calibration='sigmoid',ensemble=False,pool='mean',classes=LABELS,
        selections=selections,counts=counts,limitation=LIMITATION,input_audit=audit,
        original_hashes=original_hashes,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source},
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
        n_jobs=args.n_jobs,command=sys.argv))
    dump_new(out/'m1_partitions.json',[serialize_plan(d,p) for p in plans])
    tables=[[],[],[],[]];models=[]
    for plan in plans:
        k=plan['outer_fold'];print(f'M1 fold {k}: 180 ajustos SVM previstos',flush=True)
        result=run_fold(d,cube,pt,test[test.outer_fold==k],next(s for s in selections if s['outer_fold']==k),plan,args.n_jobs)
        for bucket,table in zip(tables,result[:4]):bucket.append(table)
        models.append(result[4])
    scored,oof,targets,refs=[pd.concat(t,ignore_index=True).sort_values('audio_id').reset_index(drop=True) for t in tables]
    require(len(scored)==1248 and scored.audio_id.is_unique,'cobertura test M1 incorrecta')
    expected=test.set_index('audio_id').loc[scored.audio_id]
    np.testing.assert_array_equal(scored[PA+PT].to_numpy(),expected[PA+PT].to_numpy())
    comparison=metrics(scored);summary,reps=paired_bootstrap(scored)
    for name,h in original_hashes.items():require(sha(out/name)==h,'artefacte previ alterat')
    for name,table in [('m1_scores.csv',scored),('m1_audio_train_oof.csv',oof),('m1_residual_targets.csv',targets),
                       ('m1_reference_residuals.csv',refs),('m1_metrics.csv',comparison),
                       ('m1_bootstrap_summary.csv',summary),('m1_bootstrap_replicas.csv',reps)]:
        table.to_csv(out/name,index=False,mode='x')
    dump_new(out/'m1_models.json',models)
    dump_new(out/'m1_audit_report.json',dict(n_test=len(scored),n_train_oof=len(oof),n_residual_targets=len(targets),
        n_reference_residuals=len(refs),n_bootstrap_valid=int(reps.valid.sum()),counts=counts,
        test_vectors='exactly_equal_B1',prior_artifacts='hashes_unchanged',limitation=LIMITATION))
    print(comparison.to_string(index=False));print(summary.to_string(index=False))


if __name__=='__main__':main()
