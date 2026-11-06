"""B0 preliminar S1. Sense --train només audita; mai sobreescriu sortides."""
import argparse
import hashlib
import itertools
import json
import platform
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emis_research.audio_wavlm_predict import (GRIDS, INNER_K, LABELS, SEED,
                                              fit_score, make_model)
from emis_research.build_manifest import COLUMNS, parse_filename
from emis_research.make_splits import build_splits, validate

ROOT = Path(__file__).resolve().parents[3]
VERSION = 'b0_preliminary_s1_qwen3_4b_v1'
OUTPUTS = ['audio_predictions.csv', 'text_predictions.csv', 'mismatch_inputs.csv',
           'mismatch_scores.csv', 'metrics_by_fold.csv', 'fold_models.json', 'run_config.json']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def align_inputs(man, splits, ids, text):
    """Exigeix cobertura exacta; ordena per splits, mai per posició implícita."""
    man, splits, text = man.copy(), splits.copy(), text.copy()
    for name, frame, key in [('manifest', man, 'audio_id'), ('splits', splits, 'audio_id'),
                             ('Qwen', text, 'text')]:
        require(not frame.isna().any().any(), f'{name}: valors absents')
        require(frame[key].is_unique, f'{name}: duplicats de {key}')
    require(len(ids) == len(set(ids)), 'embeddings: audio_id duplicats')
    require(set(ids) == set(man.audio_id) == set(splits.audio_id), 'cobertura audio_id incompatible')
    man['group_text'] = man.text_category + '_' + man.text_id
    require(man.groupby('group_text').text.nunique().eq(1).all(), 'grup amb múltiples textos')
    require(man.groupby('text').group_text.nunique().eq(1).all(), 'text en múltiples grups')
    for col in ['text_emotion', 'generation_emotion']:
        require(man[col].isin(LABELS).all(), f'etiquetes invàlides: {col}')
    shared = ['group_text', 'reference_id', 'tts_system', 'generation_emotion']
    a = man.set_index('audio_id'); b = splits.set_index('audio_id').loc[a.index]
    require(a[shared].equals(b[shared]), 'metadades splits incompatibles')
    validate(splits)
    rebuilt = build_splits(man)
    require(rebuilt.set_index('audio_id').sort_index().equals(
        splits.set_index('audio_id')[rebuilt.columns.drop('audio_id')].sort_index()),
        'assignacions de splits no reproduïbles')
    require(set(text.text) == set(man.text), 'cobertura textual incompatible')
    q = text.set_index('text').loc[man.text]
    require(np.array_equal(q.text_category.to_numpy(), man.text_category.to_numpy()) and
            np.array_equal(q.text_id.to_numpy(), man.text_id.to_numpy()), 'identitat textual incompatible')
    pc = [f'p_{c}' for c in LABELS]
    p = text[pc].to_numpy(float)
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all() and
            np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0), 'puntuacions Qwen invàlides')
    mass = text.option_mass.to_numpy(float)
    require(np.isfinite(mass).all() and (mass >= 0).all() and (mass <= 1 + 1e-6).all(),
            'option_mass invàlida')
    text['pred_text'] = np.array(LABELS)[p.argmax(axis=1)]
    text['text_model'] = 'qwen3_4b'
    d = splits[['audio_id', 'fold_S1']].merge(man, on='audio_id', validate='one_to_one')
    d = d.merge(pd.DataFrame({'audio_id': ids, 'feature_row': np.arange(len(ids))}),
                on='audio_id', validate='one_to_one')
    require(set(d.fold_S1) == set(range(5)), 'S1 ha de tenir cinc folds 0..4')
    for k in range(5):
        tr, te = d[d.fold_S1 != k], d[d.fold_S1 == k]
        require(not set(tr.group_text) & set(te.group_text), 'fuga de grup textual')
        require(not set(tr.text) & set(te.text), 'fuga de frase literal')
    return d, text, int(((p == p.max(axis=1, keepdims=True)).sum(axis=1) > 1).sum())


def load_audit(root=ROOT):
    paths = {
        'manifest': root / 'data/processed/emis_manifest.csv',
        'splits': root / 'configs/splits/splits_v1.csv',
        'splits_meta': root / 'configs/splits/splits_v1.json',
        'features': root / 'outputs/features/wavlm_base_plus_meanstd_sr16000.npz',
        'features_meta': root / 'outputs/features/wavlm_base_plus_meanstd_sr16000.json',
        'text': root / 'outputs/text_baseline/features/qwen3_4b.csv',
        'text_meta': root / 'outputs/text_baseline/run_meta.json',
        'revisions': root / 'outputs/text_baseline/model_revisions.json',
        'source_text': root / 'data/raw/text_samples_EMIS.csv',
        'source_audio': root / 'data/raw/audios_EMIS.zip',
    }
    hashes = {k: sha(v) for k, v in paths.items()}
    man = pd.read_csv(paths['manifest'], dtype=str)
    s = pd.read_csv(paths['splits'], dtype={'audio_id': str, 'reference_id': str})
    q = pd.read_csv(paths['text'], dtype={'text_id': str})
    with np.load(paths['features'], allow_pickle=False) as z:
        ids, cube = z['audio_id'].astype(str), z['mean'].astype('float64')
    d, q, ties = align_inputs(man, s, ids, q)
    require(cube.shape == (1248, 13, 768) and np.isfinite(cube).all(), 'embeddings incompatibles')
    source = pd.read_csv(paths['source_text'])
    records = []
    with zipfile.ZipFile(paths['source_audio']) as archive:
        members = sorted(n for n in archive.namelist() if n.lower().endswith('.wav'))
    require(len(members) == len(man) and set(members) == set(man.zip_member), 'ZIP incompatible')
    for member in members:
        meta = parse_filename(member)
        records.append(dict(audio_id=Path(member).stem, zip_member=member,
                            text=str(source.at[int(meta['text_id']) - 1, meta['text_category']]), **meta))
    reconstructed = pd.DataFrame(records)[COLUMNS].sort_values('audio_id').reset_index(drop=True)
    require(reconstructed.equals(man[COLUMNS].sort_values('audio_id').reset_index(drop=True)),
            'manifest incompatible amb les fonts originals')
    fm = json.loads(paths['features_meta'].read_text())
    sm = json.loads(paths['splits_meta'].read_text())
    tm = json.loads(paths['text_meta'].read_text())
    rev = json.loads(paths['revisions'].read_text())
    require(fm['features_sha256'] == hashes['features'], 'hash embeddings incompatible')
    require(tm['_global']['manifest_sha256'] == hashes['manifest'], 'hash manifest textual incompatible')
    require(tm['_global']['classes'] == LABELS, 'ordre de classes textual incompatible')
    require(tm['qwen3_4b']['revision'] == rev[tm['qwen3_4b']['repo']], 'revisió Qwen incompatible')
    report = dict(version=VERSION, hashes=hashes, n_audio=len(d), n_text=d.group_text.nunique(),
                  fold_sizes={str(k): int(v) for k, v in d.fold_S1.value_counts().sort_index().items()},
                  y_counts={str(k): int(v) for k, v in
                            (d.text_emotion != d.generation_emotion).astype(int).value_counts().items()},
                  text_argmax_ties=ties, checks='passed',
                  recorded_manifest_hashes={'features': fm['manifest_sha256'], 'splits': sm['manifest_sha256']},
                  provenance_warning='Historical manifest bytes unavailable; semantic source reconstruction and splits match. WavLM commit unknown.',
                  wavlm_model=fm['model'], wavlm_commit=fm.get('model_commit'),
                  text_model=tm['qwen3_4b'])
    return d, cube[d.feature_row.to_numpy()], q, report


def configurations():
    grid = GRIDS['SVM-RBF']
    return [('SVM-RBF', dict(zip(grid, values)), layer)
            for values in itertools.product(*grid.values()) for layer in [3, 6, 9, 12]]


def fit_outer(d, cube, k, configs, n_jobs=1):
    """El test només es consulta després de seleccionar i ajustar el model."""
    tr = np.flatnonzero(d.fold_S1.to_numpy() != k)
    te = np.flatnonzero(d.fold_S1.to_numpy() == k)
    require(len(tr) > 0 and len(te) > 0, 'fold extern buit')
    require(not set(d.iloc[tr].group_text) & set(d.iloc[te].group_text), 'fuga externa')
    x, y, g = cube[tr], d.iloc[tr].generation_emotion.to_numpy(), d.iloc[tr].group_text.to_numpy()
    require(len(np.unique(g)) >= 2, 'CV interna inviable')
    inner = list(StratifiedGroupKFold(n_splits=min(INNER_K, len(np.unique(g))),
                 shuffle=True, random_state=SEED).split(np.zeros(len(tr)), y, g))
    for a, b in inner:
        require(not set(g[a]) & set(g[b]), 'fuga interna')
        require(set(y[a]) == set(LABELS) and set(y[b]) == set(LABELS), 'classes internes incompletes')
    scores = Parallel(n_jobs=n_jobs)(delayed(fit_score)(*c, x, y, a, b)
                                      for c in configs for a, b in inner)
    means = np.asarray(scores).reshape(len(configs), len(inner)).mean(axis=1)
    best = int(means.argmax()); fam, params, layer = configs[best]
    model = make_model(fam, params).fit(x[:, layer], y)
    require(list(model.named_steps['clf'].classes_) == LABELS, 'classes_ SVM incompatibles')
    pred = model.predict(cube[te, layer])
    record = dict(outer_fold=int(k), layer=int(layer), params=params, inner_uar=float(means[best]),
                  train_audio_ids=d.iloc[tr].audio_id.tolist(), test_audio_ids=d.iloc[te].audio_id.tolist(),
                  inner_splits=[{'train_audio_ids': d.iloc[tr[a]].audio_id.tolist(),
                                 'validation_audio_ids': d.iloc[tr[b]].audio_id.tolist()} for a, b in inner])
    rows = d.iloc[te][['audio_id']].assign(protocol='S1', outer_fold=k, role='test',
              pred_audio=pred, audio_model='WavLM+SVM-RBF', calibration='none',
              model_fit_id=f'{VERSION}_fold{k}', experiment_version=VERSION)
    return rows, record, model


def score_inputs(d, acoustic, text):
    require(not acoustic.duplicated(['protocol', 'outer_fold', 'audio_id', 'role']).any(), 'prediccions duplicades')
    require(acoustic.audio_id.is_unique and set(acoustic.audio_id) == set(d.audio_id), 'cobertura acústica incompatible')
    require(acoustic.protocol.eq('S1').all() and acoustic.role.eq('test').all(), 'protocol/role incompatible')
    require(acoustic.pred_audio.isin(LABELS).all(), 'etiqueta acústica invàlida')
    require(text.text.is_unique and set(text.text) == set(d.text), 'cobertura textual incompatible')
    require(text.pred_text.isin(LABELS).all(), 'etiqueta textual invàlida')
    joined = d.merge(acoustic, on='audio_id', validate='one_to_one')
    require(joined.fold_S1.eq(joined.outer_fold).all(), 'fold acústic incompatible')
    joined = joined.merge(text[['text', 'pred_text', 'text_model', 'option_mass']], on='text', validate='many_to_one')
    joined['y_mismatch'] = (joined.text_emotion != joined.generation_emotion).astype(int)
    joined['b0_score'] = (joined.pred_audio != joined.pred_text).astype(int)
    return joined


def metrics_by_fold(scored):
    rows = []
    for k, part in scored.groupby('outer_fold', sort=True):
        y, score = part.y_mismatch.to_numpy(), part.b0_score.to_numpy()
        valid = len(np.unique(y)) == 2
        rows.append(dict(protocol='S1', outer_fold=int(k), method='B0_preliminary', n=len(y),
                         prevalence=float(y.mean()), auroc=float(roc_auc_score(y, score)) if valid else None,
                         average_precision=float(average_precision_score(y, score)) if valid else None,
                         balanced_accuracy=float(balanced_accuracy_score(y, score)) if valid else None,
                         f1=float(f1_score(y, score, zero_division=0)) if valid else None,
                         status='ok' if valid else 'not_calculated: single ground-truth class',
                         decision_rule='b0_score == 1', experiment_version=VERSION))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', action='store_true', help='Executa entrenament costós; requereix autorització prèvia')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'outputs/mismatch_v1')
    ap.add_argument('--n-jobs', type=int, default=1)
    args = ap.parse_args()
    out = args.out_dir
    require(not any((out / name).exists() for name in OUTPUTS), 'sortides existents: no se sobreescriuen')
    d, cube, text, audit = load_audit()
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / 'audit_report.json'
    if audit_path.exists():
        require(json.loads(audit_path.read_text(encoding='utf-8')) == audit, 'auditoria anterior incompatible')
    else:
        dump_new(audit_path, audit)
    print('Auditoria OK; 1605 ajustos SVM previstos.', flush=True)
    if not args.train:
        print('Sense entrenament. Cal autorització abans de --train.')
        return
    source_paths = [Path(__file__), ROOT / 'src/emis_research/audio_wavlm_predict.py',
                    ROOT / 'src/emis_research/make_splits.py', ROOT / 'src/emis_research/build_manifest.py']
    dump_new(out / 'run_config.json', dict(version=VERSION, command=sys.argv, seed=SEED,
             pool='mean', configurations=configurations(), inner_k=INNER_K, n_jobs=args.n_jobs,
             python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
             sklearn=sklearn.__version__, source_hashes={str(p.relative_to(ROOT)): sha(p) for p in source_paths},
             input_hashes=audit['hashes'], text_model=audit['text_model']))
    predictions, records = [], []
    for k in range(5):
        print(f'S1 fold {k}: selecció interna i ajust final', flush=True)
        rows, record, _ = fit_outer(d, cube, k, configurations(), args.n_jobs)
        predictions.append(rows); records.append(record)
    acoustic = pd.concat(predictions, ignore_index=True)
    scored = score_inputs(d, acoustic, text)
    acoustic.to_csv(out / 'audio_predictions.csv', index=False, mode='x')
    text.to_csv(out / 'text_predictions.csv', index=False, mode='x')
    scored.drop(columns='b0_score').to_csv(out / 'mismatch_inputs.csv', index=False, mode='x')
    scored.to_csv(out / 'mismatch_scores.csv', index=False, mode='x')
    metrics = metrics_by_fold(scored)
    metrics.to_csv(out / 'metrics_by_fold.csv', index=False, mode='x')
    dump_new(out / 'fold_models.json', records)
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
