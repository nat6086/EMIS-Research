"""Anàlisi exploratòria de scores fixos S1. Cap entrenament ni llindar binari."""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score,average_precision_score

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from emis_research.mismatch.b0_labels import ROOT,require,sha,dump_new
from emis_research.mismatch.compare_b0_b1 import strict_pair,bootstrap,summarize

KEY=['audio_id','protocol','outer_fold']
SCORES={'B0':'b0_score','B1':'b1_score','M1':'m1_score'}
PREFIX='explore_s1_'


def align(b0,b1,m1,manifest,splits):
    paired=strict_pair(b0,b1)
    require(m1.audio_id.is_unique and not m1.duplicated(KEY).any(),'M1 duplicat')
    require(m1.role.eq('test').all() and m1.protocol.eq('S1').all(),'M1 protocol/rol incorrecte')
    require(set(map(tuple,paired[KEY].to_numpy()))==set(map(tuple,m1[KEY].to_numpy())),'cobertura M1 diferent')
    merged=paired.merge(m1[KEY+['group_text','y_mismatch','m1_score']],on=KEY,
        validate='one_to_one',suffixes=('','_m1'))
    require(merged.y_mismatch.eq(merged.y_mismatch_m1).all() and merged.group_text.eq(merged.group_text_m1).all(),
            'ground truth o grup M1 diferent')
    merged=merged.drop(columns=['y_mismatch_m1','group_text_m1'])
    require(manifest.audio_id.is_unique and splits.audio_id.is_unique,'manifest/splits duplicat')
    require(set(merged.audio_id)==set(manifest.audio_id)==set(splits.audio_id),'cobertura manifest/splits diferent')
    m=manifest.set_index('audio_id').loc[merged.audio_id]
    require(np.array_equal(m.text_category+'_'+m.text_id,merged.group_text),'grup manifest diferent')
    require(np.array_equal((m.text_emotion!=m.generation_emotion).astype(int),merged.y_mismatch),'etiqueta manifest diferent')
    require(np.array_equal(splits.set_index('audio_id').loc[merged.audio_id].fold_S1,merged.outer_fold),'fold diferent')
    for c in ['text','text_cue_type','text_emotion','generation_emotion','tts_system','reference_id']:
        require(m[c].notna().all(),f'metadades absents: {c}')
        merged[c]=m[c].to_numpy()
        for frame in [b0,b1,m1]:
            actual=frame.set_index('audio_id').loc[merged.audio_id,c].astype(str)
            if c=='reference_id':
                # M1 va serialitzar la referència numèrica sense zeros inicials.
                # Es contrasta l'equivalència i es conserva el literal del manifest.
                require(actual.str.fullmatch(r'\d+').all() and merged[c].str.fullmatch(r'\d+').all(),
                        'referències no numèriques inesperades')
                actual=actual.map(int).to_numpy();expected=merged[c].map(int).to_numpy()
            else:expected=merged[c].to_numpy()
            require(np.array_equal(actual,expected),f'metadades {c} diferents')
    for col in SCORES.values():
        v=merged[col].to_numpy(float)
        require(np.isfinite(v).all() and (v>=0).all(),'scores invàlids')
    require(merged.groupby('group_text').text.nunique().eq(1).all() and
            merged.groupby('group_text').text_cue_type.nunique().eq(1).all(),'grup inconsistent')
    return merged.sort_values('audio_id').reset_index(drop=True)


def subgroup_metrics(data):
    rows=[]
    for cue,part in data.groupby('text_cue_type',sort=True):
        row=dict(text_cue_type=cue,n_audio=len(part),n_text=part.group_text.nunique(),
                 prevalence=float(part.y_mismatch.mean()),status='ok' if part.y_mismatch.nunique()==2 else 'single_class')
        for method,col in SCORES.items():
            row[f'auroc_{method}']=float(roc_auc_score(part.y_mismatch,part[col])) if row['status']=='ok' else np.nan
            row[f'ap_{method}']=float(average_precision_score(part.y_mismatch,part[col])) if row['status']=='ok' else np.nan
        for metric in ['auroc','ap']:
            row[f'{metric}_M1_minus_B1']=row[f'{metric}_M1']-row[f'{metric}_B1']
            row[f'{metric}_B1_minus_B0']=row[f'{metric}_B1']-row[f'{metric}_B0']
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_bootstrap(data,n_boot=10000):
    summaries=[];replicas=[]
    for cue,part in data.groupby('text_cue_type',sort=True):
        # No cutoff inventat de mida: >=2 grups per resampling; mida sempre visible.
        if part.group_text.nunique()<2 or part.y_mismatch.nunique()!=2:
            summaries.append(pd.DataFrame([dict(text_cue_type=cue,status='not_estimable',n_text=part.group_text.nunique())]))
            continue
        p=part[['group_text','y_mismatch','b1_score','m1_score']].rename(columns={'b1_score':'b0_score','m1_score':'b1_score'})
        print(f'Bootstrap {cue}: {part.group_text.nunique()} textos',flush=True)
        obs,r,_=bootstrap(p,n_boot=n_boot,seed=42,progress=n_boot==10000)
        mapping={'auroc_b0':'auroc_B1','auroc_b1':'auroc_M1','ap_b0':'ap_B1','ap_b1':'ap_M1'}
        if r.valid.any():
            s=summarize(obs,r);s['measure']=s.measure.map(lambda v:mapping.get(v,v))
            s=s.assign(text_cue_type=cue,n_text=part.group_text.nunique(),comparison='M1_minus_B1',status='exploratory')
        else:
            s=pd.DataFrame([dict(text_cue_type=cue,status='no_valid_replicas',n_valid=0,n_invalid=len(r))])
        summaries.append(s);replicas.append(r.rename(columns=mapping).assign(text_cue_type=cue))
    return pd.concat(summaries,ignore_index=True),pd.concat(replicas,ignore_index=True) if replicas else pd.DataFrame()


def relative_scores(data):
    d=data.copy()
    # Percentil empíric global: rang mitjà / N, mateixa referència per tots els subgrups.
    for method,col in SCORES.items():d[f'percentile_{method}']=100*d[col].rank(method='average',pct=True)
    d['percentile_gap_M1_minus_B1']=d.percentile_M1-d.percentile_B1
    d['joint_high_percentile']=d[['percentile_M1','percentile_B1']].min(axis=1)
    return d


def examples(relative,n=3):
    tables=[]
    for category,col,ascending in [('B1_higher_relative','percentile_gap_M1_minus_B1',True),
                                  ('M1_higher_relative','percentile_gap_M1_minus_B1',False),
                                  ('both_high','joint_high_percentile',False)]:
        # Un exemple per text a cada categoria; selecció determinista, no inferència.
        t=relative.sort_values([col,'audio_id'],ascending=[ascending,True]).drop_duplicates('group_text').head(n)
        tables.append(t.assign(example_category=category))
    return pd.concat(tables,ignore_index=True)


def within_text(data):
    rows=[]
    # Congruent vs cada generació incongruent dins del mateix text I TTS.
    for (group,tts),p in data.groupby(['group_text','tts_system'],sort=True):
        c=p[p.y_mismatch==0];v=p[p.y_mismatch==1]
        require(len(c)==1 and len(v)==3 and p.generation_emotion.nunique()==4,'variants text/TTS incompletes')
        c=c.iloc[0]
        for _,x in v.iterrows():
            row=dict(group_text=group,tts_system=tts,text_cue_type=x.text_cue_type,
                congruent_audio_id=c.audio_id,incongruent_audio_id=x.audio_id,
                congruent_generation_emotion=c.generation_emotion,incongruent_generation_emotion=x.generation_emotion,
                same_reference=bool(c.reference_id==x.reference_id))
            for method,col in SCORES.items():
                delta=float(x[col]-c[col]);row[f'delta_{method}']=delta
                row[f'win_{method}']=int(delta>0);row[f'tie_{method}']=int(delta==0)
                row[f'concordance_{method}']=float(delta>0)+.5*float(delta==0)
            rows.append(row)
    pairs=pd.DataFrame(rows);summaries=[]
    for cue,p in [('all',pairs)]+[(k,v) for k,v in pairs.groupby('text_cue_type',sort=True)]:
        row=dict(text_cue_type=cue,n_pairs=len(p),n_text=p.group_text.nunique(),same_reference_fraction=float(p.same_reference.mean()))
        for method in SCORES:
            row[f'wins_{method}']=int(p[f'win_{method}'].sum());row[f'ties_{method}']=int(p[f'tie_{method}'].sum())
            row[f'losses_{method}']=len(p)-row[f'wins_{method}']-row[f'ties_{method}']
            # Mitjana per text: cada text pesa igual.
            row[f'concordance_{method}']=float(p.groupby('group_text')[f'concordance_{method}'].mean().mean())
        summaries.append(row)
    return pairs,pd.DataFrame(summaries)


def main():
    out=ROOT/'outputs/mismatch_v1'
    names=['subgroup_metrics.csv','subgroup_bootstrap_summary.csv','subgroup_bootstrap_replicas.csv',
           'relative_scores.csv','examples.csv','within_text_pairs.csv','within_text_summary.csv','audit.json']
    require(not any((out/(PREFIX+n)).exists() for n in names),'sortides exploratòries existents')
    hashes={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    frames=[pd.read_csv(out/n,dtype={'reference_id':str,'text_id':str}) for n in ['mismatch_scores.csv','b1_scores.csv','m1_scores.csv']]
    manpath=ROOT/'data/processed/emis_manifest.csv';splitpath=ROOT/'configs/splits/splits_v1.csv'
    d=align(*frames,pd.read_csv(manpath,dtype=str),pd.read_csv(splitpath))
    require(len(d)==1248 and d.group_text.nunique()==104,'corpus incomplet')
    require(d.groupby('group_text').size().eq(12).all(),'mida de grup inesperada')
    metric=subgroup_metrics(d);relative=relative_scores(d);ex=examples(relative);pairs,pair_summary=within_text(d)
    print(metric.to_string(index=False),flush=True)
    summary,reps=subgroup_bootstrap(d)
    for name,h in hashes.items():require(sha(out/name)==h,'artefacte previ alterat')
    for name,t in zip(names[:-1],[metric,summary,reps,relative,ex,pairs,pair_summary]):
        t.to_csv(out/(PREFIX+name),index=False,mode='x')
    dump_new(out/(PREFIX+'audit.json'),dict(n_audio=len(d),n_text=d.group_text.nunique(),
        literal_cue_values=sorted(d.text_cue_type.unique()),seed=42,n_boot=10000,source_hashes=hashes,
        manifest_sha256=sha(manpath),splits_sha256=sha(splitpath),script_sha256=sha(__file__),
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
        percentile_rule='global average rank / N * 100; descriptive only; no decision thresholds',
        examples='3 extreme ranks per category, distinct texts within category, ties sorted by audio_id',
        pairs='same text and TTS, incongruent minus congruent; ties count 0.5; shared references reported, not enforced',
        reference_id_note='M1 CSV lost leading zeros; numeric equivalence checked against manifest; output restores manifest literal.',
        bootstrap='within subgroup, sample all its texts with replacement preserving full clusters; no replacement of invalid replicas',
        caveats='Exploratory multiple comparisons; no multiplicity adjustment or training uncertainty. Synthetic labels, not perception. B0 lacks B1 acoustic calibration.'))
    print(summary.to_string(index=False));print(pair_summary.to_string(index=False))


if __name__=='__main__':main()
