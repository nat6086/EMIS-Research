"""Jensen–Shannon base 2 i auditoria B1; no entrena en importar el mòdul."""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emis_research.mismatch.b0_labels import (LABELS, ROOT, SEED, dump_new,
    load_audit, make_model, metrics_by_fold, require, sha)

PA = [f'p_audio_{c}' for c in LABELS]
PT = [f'p_text_{c}' for c in LABELS]
KEY = ['protocol', 'outer_fold', 'audio_id', 'role']
VERSION = 'b1_s1_qwen3_4b_v1'
OUTPUTS = ['b1_run_config.json', 'b1_scores.csv', 'b1_metrics.csv', 'b1_fold_models.json',
           'b1_comparison.csv', 'b1_audit_report.json']


def load_b0(d, audit):
    directory = ROOT / 'outputs/mismatch_v1'
    files = ['run_config.json', 'fold_models.json', 'mismatch_scores.csv', 'metrics_by_fold.csv',
             'audio_predictions.csv', 'text_predictions.csv', 'audit_report.json', 'mismatch_inputs.csv']
    hashes = {f: sha(directory / f) for f in files}
    config = json.loads((directory / 'run_config.json').read_text())
    records = json.loads((directory / 'fold_models.json').read_text())
    baseline = pd.read_csv(directory / 'mismatch_scores.csv')
    require(config['input_hashes'] == audit['hashes'], 'fonts diferents de B0')
    require(config['pool'] == 'mean' and config['seed'] == SEED, 'configuració B0 incompatible')
    expected = d[['audio_id']].assign(protocol='S1', outer_fold=d.fold_S1, role='test',
               y_mismatch=(d.text_emotion != d.generation_emotion).astype(int))
    pair_with_b0(expected, baseline)
    pd.testing.assert_frame_equal(metrics_by_fold(baseline), pd.read_csv(directory / 'metrics_by_fold.csv'),
                                   check_dtype=False)
    require(len(records) == 5 and {r['outer_fold'] for r in records} == set(range(5)), 'ajustos B0 incomplets')
    for r in records:
        k = r['outer_fold']
        require(set(r['train_audio_ids']) == set(d.loc[d.fold_S1 != k, 'audio_id']) and
                set(r['test_audio_ids']) == set(d.loc[d.fold_S1 == k, 'audio_id']), 'partició B0 incompatible')
        require(['SVM-RBF', r['params'], r['layer']] in config['configurations'], 'selecció B0 desconeguda')
    return baseline, records, hashes


def calibration_plan(d, k):
    tr = np.flatnonzero(d.fold_S1.to_numpy() != k)
    te = np.flatnonzero(d.fold_S1.to_numpy() == k)
    require(len(tr) > 0 and len(te) > 0 and d.audio_id.is_unique, 'partició externa invàlida')
    require(not set(d.iloc[tr].group_text) & set(d.iloc[te].group_text), 'fuga externa')
    train = d.iloc[tr]
    y, g = train.generation_emotion.to_numpy(), train.group_text.to_numpy()
    require(len(np.unique(g)) >= 5, 'calibratge agrupat de cinc folds inviable')
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(tr)), y, g))
    for a, b in splits:
        require(not set(g[a]) & set(g[b]), 'fuga entre ajust i calibratge')
        require(set(y[a]) == set(LABELS) and set(y[b]) == set(LABELS), 'classes de calibratge incompletes')
    require(np.array_equal(np.sort(np.concatenate([b for _, b in splits])), np.arange(len(tr))),
            'cobertura OOF de calibratge invàlida')
    return tr, te, splits


def fit_calibrated(d, cube, selection, ensemble, n_jobs=1):
    require(type(ensemble) is bool, 'cal validar explícitament ensemble True/False')
    k, layer = selection['outer_fold'], selection['layer']
    tr, te, splits = calibration_plan(d, k)
    model = CalibratedClassifierCV(make_model('SVM-RBF', selection['params']),
              method='sigmoid', cv=splits, ensemble=ensemble, n_jobs=n_jobs)
    model.fit(cube[tr, layer], d.iloc[tr].generation_emotion.to_numpy())
    p = canonical_probabilities(model.predict_proba(cube[te, layer]), model.classes_)
    fit_id = f'{VERSION}_ensemble{ensemble}_fold{k}'
    rows = d.iloc[te][['audio_id']].assign(protocol='S1', outer_fold=k, role='test',
             model_fit_id=fit_id, experiment_version=VERSION,
             pred_audio_calibrated=np.array(LABELS)[p.argmax(axis=1)],
             audio_argmax_tie=(p == p.max(axis=1, keepdims=True)).sum(axis=1) > 1,
             **dict(zip(PA, p.T)))
    record = dict(model_fit_id=fit_id, outer_fold=k, layer=layer, params=selection['params'],
                  classes=list(model.classes_), method='sigmoid', ensemble=ensemble,
                  train_audio_ids=d.iloc[tr].audio_id.tolist(), test_audio_ids=d.iloc[te].audio_id.tolist(),
                  calibration_splits=[{'classifier_train_audio_ids': d.iloc[tr[a]].audio_id.tolist(),
                                       'calibration_audio_ids': d.iloc[tr[b]].audio_id.tolist()} for a,b in splits])
    return rows, record, model


def assemble_scores(d, acoustic, text, baseline):
    require(acoustic.audio_id.is_unique and set(acoustic.audio_id) == set(d.audio_id), 'cobertura acústica invàlida')
    require(text.text.is_unique and set(text.text) == set(d.text), 'cobertura textual invàlida')
    q = text[['text', 'option_mass'] + [f'p_{c}' for c in LABELS]].rename(
            columns={f'p_{c}': f'p_text_{c}' for c in LABELS})
    validate_probabilities(q[PT])
    joined = d.merge(acoustic, on='audio_id', validate='one_to_one').merge(q, on='text', validate='many_to_one')
    require(joined.outer_fold.eq(joined.fold_S1).all(), 'fold acústic incorrecte')
    joined['y_mismatch'] = (joined.text_emotion != joined.generation_emotion).astype(int)
    joined['b1_score'] = js_divergence(joined[PA], joined[PT])
    joined['pred_text'] = np.array(LABELS)[joined[PT].to_numpy().argmax(axis=1)]
    joined['text_argmax_tie'] = (joined[PT].to_numpy() == joined[PT].to_numpy().max(axis=1, keepdims=True)).sum(axis=1) > 1
    paired = pair_with_b0(joined, baseline)
    old = baseline.set_index('audio_id').loc[paired.audio_id]
    require(np.array_equal(old.pred_text.to_numpy(), paired.pred_text.to_numpy()), 'prediccions textuals diferents de B0')
    paired['audio_label_changed_vs_b0'] = paired.pred_audio_calibrated.to_numpy() != old.pred_audio.to_numpy()
    return paired


def validate_probabilities(values):
    """Valida sense retallar, renormalitzar ni aplicar softmax."""
    p = np.asarray(values, dtype=float)
    require(p.ndim == 2 and p.shape[1] == 4, 'calen quatre probabilitats per fila')
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(),
            'probabilitats no finites o fora de [0,1]')
    require(np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0), 'suma de probabilitats invàlida')
    return p


def canonical_probabilities(values, classes):
    require(len(classes) == 4 and set(classes) == set(LABELS), 'classes_ incompatibles')
    p = np.asarray(values)
    require(p.ndim == 2 and p.shape[1] == 4, 'shape de probabilitats invàlid')
    return validate_probabilities(p[:, [list(classes).index(c) for c in LABELS]])


def js_divergence(p, q):
    """JS, no la seva arrel. 0 log 0 = 0, sense epsilon."""
    p, q = validate_probabilities(p), validate_probabilities(q)
    require(p.shape == q.shape, 'vectors no alineats')
    m = (p + q) / 2
    terms = []
    for v in (p, q):
        term = np.zeros_like(v)
        mask = v > 0
        term[mask] = v[mask] * np.log2(v[mask] / m[mask])
        terms.append(term.sum(axis=1))
    result = (terms[0] + terms[1]) / 2
    require(np.isfinite(result).all() and (result >= 0).all() and (result <= 1).all(),
            'JS fora de [0,1]')
    return result


def pair_with_b0(scored, baseline):
    """Exigeix exactament la mateixa població i ground truth que B0."""
    for frame in (scored, baseline):
        require(not frame[KEY].isna().any().any(), 'claus absents')
        require(not frame.duplicated(KEY).any() and frame.audio_id.is_unique, 'scores duplicats')
        require(frame.protocol.eq('S1').all() and frame.role.eq('test').all(), 'protocol/role invàlid')
    require(set(map(tuple, scored[KEY].to_numpy())) == set(map(tuple, baseline[KEY].to_numpy())),
            'cobertura B1/B0 incompatible')
    paired = scored.merge(baseline[KEY + ['y_mismatch', 'b0_score']], on=KEY,
                          validate='one_to_one', suffixes=('', '_b0'))
    require(paired.y_mismatch.isin([0, 1]).all() and
            paired.y_mismatch.eq(paired.y_mismatch_b0).all(), 'ground truth B1/B0 incompatible')
    require(paired.b0_score.isin([0, 1]).all(), 'B0 no binari')
    return paired.drop(columns='y_mismatch_b0')


def ranking_comparison(paired):
    """Mètriques per fold i pooled; classificació B1 pendent de llindar."""
    rows = []
    parts = [(str(k), v) for k, v in paired.groupby('outer_fold', sort=True)] + [('global', paired)]
    for fold, part in parts:
        y = part.y_mismatch.to_numpy()
        for method, column in [('B0_preliminary', 'b0_score'), ('B1', 'b1_score')]:
            v = part[column].to_numpy(float)
            require(np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all(), 'score invàlid')
            valid = len(np.unique(y)) == 2
            rows.append(dict(protocol='S1', outer_fold=fold, method=method, n=len(part),
                prevalence=float(y.mean()), auroc=float(roc_auc_score(y, v)) if valid else None,
                average_precision=float(average_precision_score(y, v)) if valid else None,
                status='ok' if valid else 'not_calculated: single ground-truth class'))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--calibration-ensemble', choices=['false'],
                    help='Decisió metodològica explícita; cap valor per defecte')
    ap.add_argument('--n-jobs', type=int, default=1)
    args = ap.parse_args()
    if args.train:
        require(args.calibration_ensemble is not None, 'cal validar --calibration-ensemble abans d’entrenar')
    out = ROOT / 'outputs/mismatch_v1'
    require(not any((out / f).exists() for f in OUTPUTS), 'B1 existent: no se sobreescriu')
    d, cube, text, audit = load_audit()
    baseline, selections, b0_hashes = load_b0(d, audit)
    plans = [calibration_plan(d, k) for k in range(5)]
    print('Auditoria B0/B1 OK: 1248 àudios, 5 folds i calibratge agrupat viable.', flush=True)
    if not args.train:
        print('Sense entrenament. Sigmoid agrupat, ensemble=False validat; execució pendent d’autorització.')
        return
    ensemble = args.calibration_ensemble == 'true'
    source = [Path(__file__), ROOT / 'src/emis_research/mismatch/b0_labels.py',
              ROOT / 'src/emis_research/audio_wavlm_predict.py']
    config = dict(version=VERSION, command=sys.argv, classes=LABELS, seed=SEED, pool='mean',
                  method='sigmoid', ensemble=ensemble, calibration_folds=5, n_jobs=args.n_jobs,
                  selections=selections, input_hashes=audit['hashes'], b0_hashes=b0_hashes,
                  source_hashes={str(p.relative_to(ROOT)): sha(p) for p in source},
                  python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                  sklearn=sklearn.__version__, text_model=audit['text_model'],
                  classification_threshold=None, classification_status='pending_threshold',
                  limitation='B0 hyperparameters selected on full outer train, including calibration groups; outer test excluded.')
    dump_new(out / 'b1_run_config.json', config)
    rows, records = [], []
    for selection in sorted(selections, key=lambda r: r['outer_fold']):
        print(f"S1 fold {selection['outer_fold']}: calibratge agrupat", flush=True)
        predictions, record, _ = fit_calibrated(d, cube, selection, ensemble, args.n_jobs)
        rows.append(predictions); records.append(record)
    scored = assemble_scores(d, pd.concat(rows, ignore_index=True), text, baseline)
    comparison = ranking_comparison(scored)
    b1_metrics = comparison[comparison.method.eq('B1')].copy()
    b1_metrics['balanced_accuracy'] = np.nan
    b1_metrics['f1'] = np.nan
    b1_metrics['classification_status'] = 'pending_threshold'
    for name, expected in b0_hashes.items():
        require(sha(out / name) == expected, 'artefacte B0 alterat durant execució')
    scored.to_csv(out / 'b1_scores.csv', index=False, mode='x')
    b1_metrics.to_csv(out / 'b1_metrics.csv', index=False, mode='x')
    comparison.to_csv(out / 'b1_comparison.csv', index=False, mode='x')
    dump_new(out / 'b1_fold_models.json', records)
    dump_new(out / 'b1_audit_report.json', dict(input_audit=audit, b0_hashes=b0_hashes,
             n_test=len(scored), coverage='exact_match_B0',
             audio_argmax_ties=int(scored.audio_argmax_tie.sum()),
             text_argmax_ties_by_audio=int(scored.text_argmax_tie.sum()),
             audio_label_changes_vs_b0=int(scored.audio_label_changed_vs_b0.sum())))
    print(comparison.to_string(index=False))


if __name__ == '__main__':
    main()
