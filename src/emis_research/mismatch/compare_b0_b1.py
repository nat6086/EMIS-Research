"""Bootstrap aparellat per textos sobre scores S1 fixos, sense entrenament."""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emis_research.mismatch.b0_labels import ROOT, dump_new, require, sha

KEY = ['audio_id', 'protocol', 'outer_fold']
MEASURES = ['auroc_b0', 'auroc_b1', 'delta_auroc', 'ap_b0', 'ap_b1', 'delta_ap']
N_BOOT, SEED = 10000, 42
PREFIX = 'b0_b1_cluster_bootstrap'


def strict_pair(b0, b1):
    for name, frame, column in [('B0', b0, 'b0_score'), ('B1', b1, 'b1_score')]:
        required = KEY + ['role', 'group_text', 'y_mismatch', column]
        require(not frame[required].isna().any().any(), f'{name}: valors absents')
        require(frame.audio_id.is_unique and not frame.duplicated(KEY).any(), f'{name}: duplicats')
        require(frame.protocol.eq('S1').all() and frame.role.eq('test').all(), f'{name}: protocol/rol incorrecte')
        require(frame.y_mismatch.isin([0, 1]).all(), f'{name}: ground truth invàlid')
        v = frame[column].to_numpy(float)
        require(np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all(), f'{name}: scores invàlids')
    require(b0.b0_score.isin([0, 1]).all(), 'B0 no binari')
    require(set(map(tuple, b0[KEY].to_numpy())) == set(map(tuple, b1[KEY].to_numpy())), 'claus no coincidents')
    paired = b0[KEY + ['group_text', 'y_mismatch', 'b0_score']].merge(
        b1[KEY + ['group_text', 'y_mismatch', 'b1_score']], on=KEY,
        validate='one_to_one', suffixes=('', '_b1'))
    require(paired.group_text.eq(paired.group_text_b1).all(), 'grups B0/B1 diferents')
    require(paired.y_mismatch.eq(paired.y_mismatch_b1).all(), 'ground truth B0/B1 diferent')
    require(paired.groupby('group_text').outer_fold.nunique().eq(1).all(), 'grup en múltiples folds')
    return paired.drop(columns=['group_text_b1', 'y_mismatch_b1']).sort_values('audio_id').reset_index(drop=True)


def metric_vector(y, b0, b1):
    a0, a1 = roc_auc_score(y, b0), roc_auc_score(y, b1)
    p0, p1 = average_precision_score(y, b0), average_precision_score(y, b1)
    return np.array([a0, a1, a1-a0, p0, p1, p1-p0])


def sample_indices(by_group, draws):
    """Concatenació explícita: conserva tots els clips i grups repetits."""
    return np.concatenate([by_group[int(i)] for i in draws])


def bootstrap(paired, n_boot=N_BOOT, seed=SEED, progress=False):
    require(isinstance(n_boot, int) and n_boot > 0, 'nombre de rèpliques invàlid')
    require(len(paired) > 0 and paired.group_text.notna().all(), 'grups absents')
    groups = sorted(paired.group_text.unique())
    by_group = [np.flatnonzero(paired.group_text.to_numpy() == g) for g in groups]
    y, b0, b1 = (paired[c].to_numpy() for c in ['y_mismatch', 'b0_score', 'b1_score'])
    require(len(np.unique(y)) == 2, 'mostra observada sense dues classes')
    observed = metric_vector(y, b0, b1)
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(n_boot):
        draws = rng.choice(len(groups), size=len(groups), replace=True)
        idx = sample_indices(by_group, draws)
        valid = len(np.unique(y[idx])) == 2
        values = metric_vector(y[idx], b0[idx], b1[idx]) if valid else [np.nan]*6
        rows.append(dict(replica=rep, valid=valid, reason='' if valid else 'single_ground_truth_class',
                         n_audio=len(idx), n_unique_groups=len(np.unique(draws)),
                         **dict(zip(MEASURES, values))))
        if progress and (rep+1) % 1000 == 0:
            print(f'Bootstrap {rep+1}/{n_boot}', flush=True)
    return observed, pd.DataFrame(rows), groups


def summarize(observed, replicas):
    valid = replicas.loc[replicas.valid, MEASURES]
    require(len(valid) > 0 and np.isfinite(valid.to_numpy()).all(), 'cap rèplica vàlida o mètriques no finites')
    lo, hi = np.percentile(valid.to_numpy(), [2.5, 97.5], axis=0, method='linear')
    return pd.DataFrame([dict(measure=m, observed=float(observed[i]), ci95_low=float(lo[i]),
        ci95_high=float(hi[i]), includes_zero=bool(lo[i] <= 0 <= hi[i]) if m.startswith('delta') else None,
        n_replicas=len(replicas), n_valid=len(valid), n_invalid=len(replicas)-len(valid))
        for i,m in enumerate(MEASURES)])


def main():
    out = ROOT / 'outputs/mismatch_v1'
    destinations = [out / f'{PREFIX}_{suffix}' for suffix in ['summary.csv', 'replicas.csv', 'audit.json']]
    require(not any(p.exists() for p in destinations), 'sortides existents: no se sobreescriuen')
    # Hash de tots els artefactes previs per verificar que es conserven intactes.
    originals = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    b0 = pd.read_csv(out / 'mismatch_scores.csv')
    b1 = pd.read_csv(out / 'b1_scores.csv')
    paired = strict_pair(b0,b1)
    require(len(paired) == 1248 and paired.group_text.nunique() == 104, 'corpus incomplet')
    require(paired.groupby('group_text').size().eq(12).all(), 'grup sense 12 àudios')
    require(set(paired.outer_fold) == set(range(5)), 'folds S1 incomplets')
    manifest_path = ROOT / 'data/processed/emis_manifest.csv'
    splits_path = ROOT / 'configs/splits/splits_v1.csv'
    manifest = pd.read_csv(manifest_path, dtype=str).set_index('audio_id')
    splits = pd.read_csv(splits_path).set_index('audio_id')
    require(manifest.index.is_unique and splits.index.is_unique and
            set(paired.audio_id) == set(manifest.index) == set(splits.index), 'cobertura manifest/splits incompatible')
    m = manifest.loc[paired.audio_id]
    require(np.array_equal((m.text_category+'_'+m.text_id).to_numpy(), paired.group_text), 'grups manifest diferents')
    require(np.array_equal((m.text_emotion != m.generation_emotion).astype(int), paired.y_mismatch), 'ground truth manifest diferent')
    require(np.array_equal(splits.loc[paired.audio_id].fold_S1, paired.outer_fold), 'splits diferents')
    observed = metric_vector(paired.y_mismatch, paired.b0_score, paired.b1_score)
    stored = pd.read_csv(out / 'b1_comparison.csv', dtype={'outer_fold':str})
    global_rows = stored[stored.outer_fold.eq('global')].set_index('method')
    expected = [global_rows.loc['B0_preliminary','auroc'], global_rows.loc['B1','auroc'],
                global_rows.loc['B0_preliminary','average_precision'], global_rows.loc['B1','average_precision']]
    np.testing.assert_allclose(observed[[0,1,3,4]], expected, atol=1e-12, rtol=0)
    print('Aparellament i mètriques globals verificats: 1248 àudios, 104 textos.', flush=True)
    observed, replicas, groups = bootstrap(paired, progress=True)
    summary = summarize(observed,replicas)
    for name,h in originals.items():
        require(sha(out/name) == h, f'artefacte original alterat: {name}')
    summary.to_csv(destinations[0], index=False, mode='x')
    replicas.to_csv(destinations[1], index=False, mode='x')
    dump_new(destinations[2], dict(seed=SEED,n_boot=N_BOOT,n_valid=int(replicas.valid.sum()),
        n_invalid=int((~replicas.valid).sum()), n_audio=len(paired), group_order=groups,
        group_sizes=paired.groupby('group_text').size().to_dict(), resampling='104 groups with replacement; concatenate all clips including multiplicity',
        pairing_keys=KEY, scope='pooled S1 external test scores; fitted models fixed; no retraining',
        interval='percentile 2.5/97.5, numpy method=linear', rng='numpy.default_rng PCG64',
        original_hashes=originals, manifest_sha256=sha(manifest_path),splits_sha256=sha(splits_path),
        script_sha256=sha(__file__),python=platform.python_version(),numpy=np.__version__,
        pandas=pd.__version__,sklearn=sklearn.__version__,
        limitation='B1 adds acoustic calibration; improvement cannot be attributed only to JS. Synthetic generation mismatch, not human perception. Conditional uncertainty of fixed OOF scores, not training variability.'))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
