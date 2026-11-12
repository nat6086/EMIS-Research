"""B0 preliminar S1. Sense --train només audita; mai sobreescriu sortides.

Idea general del baseline B0 (detecció de mismatch text-emoció vs. àudio-emoció):
  1. Àudio: embeddings WavLM (mitjana per capa) -> SVM-RBF prediu l'emoció amb què
     s'ha generat l'àudio TTS (`generation_emotion`).
  2. Text: Qwen3-4B (zero-shot) dóna probabilitats per emoció; `pred_text` = argmax.
  3. Score B0: hi ha mismatch si l'emoció predita per l'àudio difereix de la del text.
  4. Avaluació: es compara amb l'etiqueta real `text_emotion != generation_emotion`.
Protocol S1: 5 folds externs, agrupats per text (cap frase ni grup textual compartit
entre train i test).
"""
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

# Permet executar el fitxer directament (python fitxer.py) afegint l'arrel del
# paquet al path; si s'executa com a mòdul (__package__ definit) no cal.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Utilitats compartides amb el pipeline d'àudio i la construcció de dades.
from emis_research.audio_wavlm_predict import (GRIDS, INNER_K, LABELS, SEED,
                                              fit_score, make_model)
from emis_research.build_manifest import COLUMNS, parse_filename
from emis_research.make_splits import build_splits, validate

ROOT = Path(__file__).resolve().parents[3]
# Identificador de la versió de l'experiment; es propaga a totes les sortides.
VERSION = 'b0_preliminary_s1_qwen3_4b_v1'
# Fitxers de sortida protegits contra sobreescriptura (audit_report.json es gestiona a part).
OUTPUTS = ['audio_predictions.csv', 'text_predictions.csv', 'mismatch_inputs.csv',
           'mismatch_scores.csv', 'metrics_by_fold.csv', 'fold_models.json', 'run_config.json']


def require(ok, message):
    """Substitut d'`assert` que no s'elimina amb `python -O`: llança ValueError si falla."""
    if not ok:
        raise ValueError(message)


def sha(path):
    """SHA-256 d'un fitxer (llegit sencer a memòria)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_new(path, value):
    """Escriu JSON amb mode 'x': falla si el fitxer ja existeix (mai sobreescriu).
    `allow_nan=False` evita generar JSON no estàndard amb NaN/Infinity."""
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def align_inputs(man, splits, ids, text):
    """Exigeix cobertura exacta; ordena per splits, mai per posició implícita.

    Valida i alinea manifest, splits, IDs dels embeddings i puntuacions de Qwen.
    Retorna (dades alineades, taula de text amb prediccions, nombre d'empats d'argmax).
    """
    # Còpies per no modificar els originals de qui crida.
    man, splits, text = man.copy(), splits.copy(), text.copy()

    # Sense valors absents i amb claus úniques a cada taula.
    for name, frame, key in [('manifest', man, 'audio_id'), ('splits', splits, 'audio_id'),
                             ('Qwen', text, 'text')]:
        require(not frame.isna().any().any(), f'{name}: valors absents')
        require(frame[key].is_unique, f'{name}: duplicats de {key}')
    require(len(ids) == len(set(ids)), 'embeddings: audio_id duplicats')
    # Els tres orígens han de cobrir exactament el mateix conjunt d'àudios.
    require(set(ids) == set(man.audio_id) == set(splits.audio_id), 'cobertura audio_id incompatible')

    # Grup textual = categoria + id de text. Ha de ser una correspondència 1:1 amb la frase.
    man['group_text'] = man.text_category + '_' + man.text_id
    require(man.groupby('group_text').text.nunique().eq(1).all(), 'grup amb múltiples textos')
    require(man.groupby('text').group_text.nunique().eq(1).all(), 'text en múltiples grups')

    # Totes les etiquetes d'emoció han de pertànyer al conjunt LABELS.
    for col in ['text_emotion', 'generation_emotion']:
        require(man[col].isin(LABELS).all(), f'etiquetes invàlides: {col}')

    # Les metadades compartides entre manifest i splits han de coincidir (alineant per audio_id).
    shared = ['group_text', 'reference_id', 'tts_system', 'generation_emotion']
    a = man.set_index('audio_id'); b = splits.set_index('audio_id').loc[a.index]
    require(a[shared].equals(b[shared]), 'metadades splits incompatibles')
    validate(splits)  # validació pròpia dels splits (definida a make_splits)

    # Reproductibilitat: reconstruir els splits des del manifest ha de donar els mateixos.
    rebuilt = build_splits(man)
    require(rebuilt.set_index('audio_id').sort_index().equals(
        splits.set_index('audio_id')[rebuilt.columns.drop('audio_id')].sort_index()),
        'assignacions de splits no reproduïbles')

    # Cobertura i identitat de les prediccions textuals respecte al manifest.
    require(set(text.text) == set(man.text), 'cobertura textual incompatible')
    q = text.set_index('text').loc[man.text]  # reordena Qwen seguint l'ordre del manifest
    require(np.array_equal(q.text_category.to_numpy(), man.text_category.to_numpy()) and
            np.array_equal(q.text_id.to_numpy(), man.text_id.to_numpy()), 'identitat textual incompatible')

    # Les probabilitats de Qwen han de ser finites, dins [0,1] i sumar 1 per fila.
    pc = [f'p_{c}' for c in LABELS]
    p = text[pc].to_numpy(float)
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all() and
            np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0), 'puntuacions Qwen invàlides')
    # option_mass: massa de probabilitat original assignada a les opcions vàlides (ha d'estar a [0,1]).
    mass = text.option_mass.to_numpy(float)
    require(np.isfinite(mass).all() and (mass >= 0).all() and (mass <= 1 + 1e-6).all(),
            'option_mass invàlida')

    # Predicció textual = classe de màxima probabilitat.
    # Nota: en cas d'empat, argmax tria la primera etiqueta de LABELS (biaix sistemàtic).
    text['pred_text'] = np.array(LABELS)[p.argmax(axis=1)]
    text['text_model'] = 'qwen3_4b'

    # Taula de treball: fold S1 + manifest + fila corresponent dins del cub d'embeddings.
    d = splits[['audio_id', 'fold_S1']].merge(man, on='audio_id', validate='one_to_one')
    d = d.merge(pd.DataFrame({'audio_id': ids, 'feature_row': np.arange(len(ids))}),
                on='audio_id', validate='one_to_one')
    require(set(d.fold_S1) == set(range(5)), 'S1 ha de tenir cinc folds 0..4')

    # Anti-fuga: cap grup textual ni frase literal pot aparèixer a train i test alhora.
    for k in range(5):
        tr, te = d[d.fold_S1 != k], d[d.fold_S1 == k]
        require(not set(tr.group_text) & set(te.group_text), 'fuga de grup textual')
        require(not set(tr.text) & set(te.text), 'fuga de frase literal')

    # Comptem files de Qwen amb més d'una classe amb la probabilitat màxima (empats exactes).
    return d, text, int(((p == p.max(axis=1, keepdims=True)).sum(axis=1) > 1).sum())


def load_audit(root=ROOT):
    """Carrega totes les entrades, calcula hashes, les valida contra les fonts
    originals i genera l'informe d'auditoria.

    Retorna (taula alineada, embeddings ordenats com la taula, text amb prediccions, informe).
    """
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
    # Hash de cada entrada (traçabilitat de la prova).
    hashes = {k: sha(v) for k, v in paths.items()}

    # dtype=str per no perdre zeros inicials als identificadors.
    man = pd.read_csv(paths['manifest'], dtype=str)
    s = pd.read_csv(paths['splits'], dtype={'audio_id': str, 'reference_id': str})
    q = pd.read_csv(paths['text'], dtype={'text_id': str})
    # allow_pickle=False: només arrays numèrics, més segur.
    with np.load(paths['features'], allow_pickle=False) as z:
        ids, cube = z['audio_id'].astype(str), z['mean'].astype('float64')
    d, q, ties = align_inputs(man, s, ids, q)
    # Cub esperat: 1248 àudios x 13 capes (embedding + 12 transformers) x 768 dimensions.
    require(cube.shape == (1248, 13, 768) and np.isfinite(cube).all(), 'embeddings incompatibles')

    # Reconstrucció independent del manifest a partir del ZIP d'àudio i el CSV de textos originals.
    source = pd.read_csv(paths['source_text'])
    records = []
    with zipfile.ZipFile(paths['source_audio']) as archive:
        members = sorted(n for n in archive.namelist() if n.lower().endswith('.wav'))
    require(len(members) == len(man) and set(members) == set(man.zip_member), 'ZIP incompatible')
    for member in members:
        meta = parse_filename(member)  # extreu metadades del nom del fitxer
        records.append(dict(audio_id=Path(member).stem, zip_member=member,
                            # text_id és 1-indexat; la columna és la categoria de text.
                            text=str(source.at[int(meta['text_id']) - 1, meta['text_category']]), **meta))
    reconstructed = pd.DataFrame(records)[COLUMNS].sort_values('audio_id').reset_index(drop=True)
    require(reconstructed.equals(man[COLUMNS].sort_values('audio_id').reset_index(drop=True)),
            'manifest incompatible amb les fonts originals')

    # Metadades registrades pels passos previs: han de ser coherents amb els fitxers actuals.
    fm = json.loads(paths['features_meta'].read_text())
    sm = json.loads(paths['splits_meta'].read_text())
    tm = json.loads(paths['text_meta'].read_text())
    rev = json.loads(paths['revisions'].read_text())
    require(fm['features_sha256'] == hashes['features'], 'hash embeddings incompatible')
    require(tm['_global']['manifest_sha256'] == hashes['manifest'], 'hash manifest textual incompatible')
    require(tm['_global']['classes'] == LABELS, 'ordre de classes textual incompatible')
    require(tm['qwen3_4b']['revision'] == rev[tm['qwen3_4b']['repo']], 'revisió Qwen incompatible')

    # Informe d'auditoria (es desa a audit_report.json).
    report = dict(version=VERSION, hashes=hashes, n_audio=len(d), n_text=d.group_text.nunique(),
                  fold_sizes={str(k): int(v) for k, v in d.fold_S1.value_counts().sort_index().items()},
                  # Distribució de l'etiqueta objectiu: 1 = mismatch, 0 = coherent.
                  y_counts={str(k): int(v) for k, v in
                            (d.text_emotion != d.generation_emotion).astype(int).value_counts().items()},
                  text_argmax_ties=ties, checks='passed',
                  recorded_manifest_hashes={'features': fm['manifest_sha256'], 'splits': sm['manifest_sha256']},
                  provenance_warning='Historical manifest bytes unavailable; semantic source reconstruction and splits match. WavLM commit unknown.',
                  wavlm_model=fm['model'], wavlm_commit=fm.get('model_commit'),
                  text_model=tm['qwen3_4b'])
    # Els embeddings es reordenen per seguir l'ordre de `d` (via feature_row).
    return d, cube[d.feature_row.to_numpy()], q, report


def configurations():
    """Llista de configuracions candidates: (família, hiperparàmetres, capa WavLM).
    Producte cartesià del grid de l'SVM-RBF per les capes 3, 6, 9 i 12."""
    grid = GRIDS['SVM-RBF']
    return [('SVM-RBF', dict(zip(grid, values)), layer)
            for values in itertools.product(*grid.values()) for layer in [3, 6, 9, 12]]


def fit_outer(d, cube, k, configs, n_jobs=1):
    """El test només es consulta després de seleccionar i ajustar el model.

    CV niuada: dins del train del fold extern `k` es tria la millor configuració
    amb CV interna agrupada; després s'ajusta al train complet i es prediu el test.
    Retorna (prediccions del test, registre de l'ajust, model).
    """
    # Índexs posicionals de train (tots els altres folds) i test (fold k).
    tr = np.flatnonzero(d.fold_S1.to_numpy() != k)
    te = np.flatnonzero(d.fold_S1.to_numpy() == k)
    require(len(tr) > 0 and len(te) > 0, 'fold extern buit')
    require(not set(d.iloc[tr].group_text) & set(d.iloc[te].group_text), 'fuga externa')
    # x: embeddings de train; y: etiqueta d'emoció d'àudio; g: grups per a la CV agrupada.
    x, y, g = cube[tr], d.iloc[tr].generation_emotion.to_numpy(), d.iloc[tr].group_text.to_numpy()
    require(len(np.unique(g)) >= 2, 'CV interna inviable')

    # Splits interns estratificats per classe i agrupats per text (determinista via SEED).
    inner = list(StratifiedGroupKFold(n_splits=min(INNER_K, len(np.unique(g))),
                 shuffle=True, random_state=SEED).split(np.zeros(len(tr)), y, g))
    for a, b in inner:
        require(not set(g[a]) & set(g[b]), 'fuga interna')
        require(set(y[a]) == set(LABELS) and set(y[b]) == set(LABELS), 'classes internes incompletes')

    # Avalua totes les configuracions x splits interns en paral·lel (UAR a validació).
    scores = Parallel(n_jobs=n_jobs)(delayed(fit_score)(*c, x, y, a, b)
                                      for c in configs for a, b in inner)
    # Mitjana per configuració sobre els splits interns; es tria la millor
    # (en cas d'empat, argmax retorna la primera).
    means = np.asarray(scores).reshape(len(configs), len(inner)).mean(axis=1)
    best = int(means.argmax()); fam, params, layer = configs[best]

    # Ajust final amb tot el train, només amb la capa seleccionada.
    model = make_model(fam, params).fit(x[:, layer], y)
    require(list(model.named_steps['clf'].classes_) == LABELS, 'classes_ SVM incompatibles')
    # Només ara es toca el test.
    pred = model.predict(cube[te, layer])

    # Registre complet per reproduir la selecció i els splits (train/test/interns).
    record = dict(outer_fold=int(k), layer=int(layer), params=params, inner_uar=float(means[best]),
                  train_audio_ids=d.iloc[tr].audio_id.tolist(), test_audio_ids=d.iloc[te].audio_id.tolist(),
                  inner_splits=[{'train_audio_ids': d.iloc[tr[a]].audio_id.tolist(),
                                 'validation_audio_ids': d.iloc[tr[b]].audio_id.tolist()} for a, b in inner])
    rows = d.iloc[te][['audio_id']].assign(protocol='S1', outer_fold=k, role='test',
              pred_audio=pred, audio_model='WavLM+SVM-RBF', calibration='none',
              model_fit_id=f'{VERSION}_fold{k}', experiment_version=VERSION)
    return rows, record, model


def score_inputs(d, acoustic, text):
    """Uneix prediccions acústiques i textuals i calcula l'etiqueta real i el score B0."""
    # Comprovacions de qualitat de les prediccions acústiques.
    require(not acoustic.duplicated(['protocol', 'outer_fold', 'audio_id', 'role']).any(), 'prediccions duplicades')
    require(acoustic.audio_id.is_unique and set(acoustic.audio_id) == set(d.audio_id), 'cobertura acústica incompatible')
    require(acoustic.protocol.eq('S1').all() and acoustic.role.eq('test').all(), 'protocol/role incompatible')
    require(acoustic.pred_audio.isin(LABELS).all(), 'etiqueta acústica invàlida')
    require(text.text.is_unique and set(text.text) == set(d.text), 'cobertura textual incompatible')
    require(text.pred_text.isin(LABELS).all(), 'etiqueta textual invàlida')

    joined = d.merge(acoustic, on='audio_id', validate='one_to_one')
    # El fold on s'ha predit ha de coincidir amb el fold assignat pels splits.
    require(joined.fold_S1.eq(joined.outer_fold).all(), 'fold acústic incompatible')
    # Cada text (frase) pot aparèixer en diversos àudios: unió molts-a-un.
    joined = joined.merge(text[['text', 'pred_text', 'text_model', 'option_mass']], on='text', validate='many_to_one')
    # Etiqueta real: hi ha mismatch si l'emoció del text i la de generació de l'àudio difereixen.
    joined['y_mismatch'] = (joined.text_emotion != joined.generation_emotion).astype(int)
    # Score B0: 1 si l'àudio i el text prediuen emocions diferents (score binari).
    joined['b0_score'] = (joined.pred_audio != joined.pred_text).astype(int)
    return joined


def metrics_by_fold(scored):
    """Mètriques de detecció de mismatch per fold extern."""
    rows = []
    for k, part in scored.groupby('outer_fold', sort=True):
        y, score = part.y_mismatch.to_numpy(), part.b0_score.to_numpy()
        # Les mètriques només tenen sentit si el fold conté les dues classes reals.
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
    """Punt d'entrada: audita sempre; entrena i desa resultats només amb --train."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', action='store_true', help='Executa entrenament costós; requereix autorització prèvia')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'outputs/mismatch_v1')
    ap.add_argument('--n-jobs', type=int, default=1)
    args = ap.parse_args()
    out = args.out_dir
    # Protecció: si ja existeix qualsevol sortida, s'atura abans de fer res.
    require(not any((out / name).exists() for name in OUTPUTS), 'sortides existents: no se sobreescriuen')

    # Auditoria completa de les entrades.
    d, cube, text, audit = load_audit()
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / 'audit_report.json'
    # Si ja hi ha un informe previ, ha de coincidir exactament amb l'actual; si no, es crea.
    if audit_path.exists():
        require(json.loads(audit_path.read_text(encoding='utf-8')) == audit, 'auditoria anterior incompatible')
    else:
        dump_new(audit_path, audit)
    # 1605 ajustos = 5 folds x (configuracions x splits interns + 1 ajust final) (valor deduït del grid actual).
    print('Auditoria OK; 1605 ajustos SVM previstos.', flush=True)
    if not args.train:
        print('Sense entrenament. Cal autorització abans de --train.')
        return

    # Hash del codi font rellevant, per saber exactament amb quin codi s'ha executat.
    source_paths = [Path(__file__), ROOT / 'src/emis_research/audio_wavlm_predict.py',
                    ROOT / 'src/emis_research/make_splits.py', ROOT / 'src/emis_research/build_manifest.py']
    # Configuració de l'execució, escrita ABANS d'entrenar (si falla, les sortides queden bloquejades).
    dump_new(out / 'run_config.json', dict(version=VERSION, command=sys.argv, seed=SEED,
             pool='mean', configurations=configurations(), inner_k=INNER_K, n_jobs=args.n_jobs,
             python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
             sklearn=sklearn.__version__, source_hashes={str(p.relative_to(ROOT)): sha(p) for p in source_paths},
             input_hashes=audit['hashes'], text_model=audit['text_model']))

    # Entrenament niuat: un ajust extern per fold; els models es descarten (`_`).
    predictions, records = [], []
    for k in range(5):
        print(f'S1 fold {k}: selecció interna i ajust final', flush=True)
        rows, record, _ = fit_outer(d, cube, k, configurations(), args.n_jobs)
        predictions.append(rows); records.append(record)

    # Combina prediccions, calcula scores i desa totes les sortides (mode='x': no sobreescriu).
    acoustic = pd.concat(predictions, ignore_index=True)
    scored = score_inputs(d, acoustic, text)
    acoustic.to_csv(out / 'audio_predictions.csv', index=False, mode='x')
    text.to_csv(out / 'text_predictions.csv', index=False, mode='x')
    # mismatch_inputs: mateixa taula sense el score B0 (només entrades del detector).
    scored.drop(columns='b0_score').to_csv(out / 'mismatch_inputs.csv', index=False, mode='x')
    scored.to_csv(out / 'mismatch_scores.csv', index=False, mode='x')
    metrics = metrics_by_fold(scored)
    metrics.to_csv(out / 'metrics_by_fold.csv', index=False, mode='x')
    dump_new(out / 'fold_models.json', records)
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()