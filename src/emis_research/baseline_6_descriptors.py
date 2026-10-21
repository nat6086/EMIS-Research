"""Baseline trivial amb descriptors de senyal + selecció de model amb CV aniuada.

Objectiu: quantificar quanta part de generation_emotion s'explica només amb durada,
nivell i silencis, sota els splits S1/S2/S3. És un llindar de referència, no un model.

Selecció de model SENSE usar el test: per a cada fold extern, els hiperparàmetres i
la família guanyadora es trien amb una CV interna només sobre el train d'aquell fold
(agrupada segons el protocol: S1 per text, S2 per referència, S3 per TTS). El fold
de test s'usa una sola vegada per avaluar. Només lectura: no modifica dades ni splits.
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
FEATS = ["duration_s", "rms_dbfs", "low_energy_frac", "lead_sil_s", "trail_sil_s", "peak"]
SEED, INNER_K = 42, 5
GROUP_COL = {"S1": "group_text", "S2": "reference_id", "S3": "tts_system"}
FAMILIES = ["LogReg", "RandomForest", "SVM-RBF", "HistGB"]  # ordre = desempat

GRIDS = {
    "LogReg": {"C": [0.01, 0.1, 1, 10, 100]},
    "RandomForest": {"max_depth": [4, 8, 12], "min_samples_leaf": [2, 5, 10],
                     "max_features": ["sqrt", 1.0]},
    "SVM-RBF": {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]},
    "HistGB": {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [3, 7, 15],
               "l2_regularization": [0, 1]},
}
QUICK = {"LogReg": {"C": [0.1, 1]}, "RandomForest": {"max_depth": [4], "min_samples_leaf": [5],
         "max_features": ["sqrt"]}, "SVM-RBF": {"C": [1], "gamma": [0.1]},
         "HistGB": {"learning_rate": [0.1], "max_leaf_nodes": [3], "l2_regularization": [0]}}


def make_model(family: str, p: dict):
    if family == "LogReg":
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", **p)
    elif family == "RandomForest":
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=SEED, n_jobs=1, **p)
    elif family == "SVM-RBF":
        clf = SVC(kernel="rbf", class_weight="balanced", **p)
    else:
        clf = HistGradientBoostingClassifier(max_iter=100, early_stopping=False,
                                             class_weight="balanced", random_state=SEED, **p)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])  # scaler només amb train


def expand(grids: dict) -> list[tuple[str, dict]]:
    out = []
    for fam in FAMILIES:
        keys = list(grids[fam])
        out += [(fam, dict(zip(keys, v))) for v in itertools.product(*grids[fam].values())]
    return out


def within_text_eta2(d: pd.DataFrame, col: str) -> float:
    """eta² de l'emoció després de restar la mitjana de cada text (dins d'un TTS)."""
    r = d[col] - d.groupby("group_text")[col].transform("mean")
    g = r.groupby(d["generation_emotion"])
    ss_b = (g.count() * (g.mean() - r.mean()) ** 2).sum()
    ss_t = ((r - r.mean()) ** 2).sum()
    return float(ss_b / ss_t) if ss_t > 0 else float("nan")


def inner_splits(dtr: pd.DataFrame, name: str):
    groups = dtr[GROUP_COL[name]].to_numpy()
    k = min(INNER_K, len(np.unique(groups)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    return list(sgkf.split(dtr[FEATS], dtr["generation_emotion"], groups))


def fit_score(family, params, X, y, tr, va) -> float:
    m = make_model(family, params).fit(X[tr], y[tr])
    return balanced_accuracy_score(y[va], m.predict(X[va]))


def nested_protocol(d: pd.DataFrame, name: str, configs, n_jobs: int):
    fold = d[f"fold_{name}"].to_numpy()
    y_all = d["generation_emotion"].to_numpy()
    preds = {f: np.empty(len(d), dtype=object) for f in FAMILIES + ["SELECTED"]}
    records = []
    for k in np.unique(fold):
        tr_idx, te_idx = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        dtr = d.iloc[tr_idx]
        X, y = dtr[FEATS].to_numpy(), dtr["generation_emotion"].to_numpy()
        splits = inner_splits(dtr, name)
        jobs = [(ci, si) for ci in range(len(configs)) for si in range(len(splits))]
        res = Parallel(n_jobs=n_jobs)(
            delayed(fit_score)(configs[ci][0], configs[ci][1], X, y, *splits[si])
            for ci, si in jobs)
        inner = np.zeros((len(configs), len(splits)))
        for (ci, si), r in zip(jobs, res):
            inner[ci, si] = r
        inner_mean = inner.mean(axis=1)
        best = {}
        for fam in FAMILIES:  # millor configuració de cada família, només amb el train
            cis = [i for i, c in enumerate(configs) if c[0] == fam]
            best[fam] = cis[int(np.argmax(inner_mean[cis]))]
        winner = max(FAMILIES, key=lambda f: (inner_mean[best[f]], -FAMILIES.index(f)))
        Xte, yte = d.iloc[te_idx][FEATS].to_numpy(), y_all[te_idx]
        for fam in FAMILIES:
            fam_, p = configs[best[fam]]
            m = make_model(fam_, p).fit(X, y)
            preds[fam][te_idx] = m.predict(Xte)
            if fam == winner:
                preds["SELECTED"][te_idx] = preds[fam][te_idx]
            records.append({"protocol": name, "fold": int(k), "family": fam,
                            "params": str(p), "inner_uar": inner_mean[best[fam]],
                            "test_uar": balanced_accuracy_score(yte, preds[fam][te_idx]),
                            "selected": fam == winner})
    return preds, pd.DataFrame(records)


def cluster_ci(d: pd.DataFrame, y, pred, n_boot: int):
    rng = np.random.default_rng(SEED)
    groups = d["group_text"].unique()
    by_g = {g: np.flatnonzero((d["group_text"] == g).to_numpy()) for g in groups}
    boots = [balanced_accuracy_score(y[i], pred[i]) for i in
             (np.concatenate([by_g[g] for g in rng.choice(groups, len(groups))])
              for _ in range(n_boot))]
    return np.percentile(boots, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", default=ROOT / "outputs/qc/audio_qc.csv")
    ap.add_argument("--splits", default=ROOT / "configs/splits/splits_v1.csv")
    ap.add_argument("--out", default=ROOT / "outputs/qc/shortcuts_nested.csv")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="graella reduïda, només per provar")
    ap.add_argument("--expect-n", type=int, default=1248)
    a = ap.parse_args()
    q = pd.read_csv(a.qc, dtype={"audio_id": str, "reference_id": str})
    s = pd.read_csv(a.splits, dtype={"audio_id": str, "reference_id": str})
    need = ["audio_id", "tts_system", "generation_emotion", "flag_long_edge_sil"] + FEATS
    d = q[need].merge(s[["audio_id", "group_text", "reference_id", "fold_S1", "fold_S2", "fold_S3"]],
                      on="audio_id", validate="one_to_one")
    assert len(d) == a.expect_n and d[FEATS].notna().all().all()
    configs = expand(QUICK if a.quick else GRIDS)
    print(f"sklearn {sklearn.__version__} | configuracions: {len(configs)} "
          f"({'quick' if a.quick else 'graella completa'}) | CV interna k<={INNER_K}")

    print("\nSilenci de vora: fracció amb bandera per TTS x emoció")
    print(d.groupby(["tts_system", "generation_emotion"])["flag_long_edge_sil"].mean().round(3).unstack().to_string())
    print("\neta² de l'emoció DINS de text, per TTS:")
    for col in ["duration_s", "rms_dbfs", "lead_sil_s", "low_energy_frac"]:
        print(f"  {col:16s}", {t: round(within_text_eta2(g, col), 3) for t, g in d.groupby("tts_system")})

    y = d["generation_emotion"].to_numpy()
    all_rec = []
    print(f"\nCV aniuada (atzar = 0.25). UAR agrupat de les prediccions externes + IC95 cluster bootstrap per text")
    for name, what in [("S1", "textos nous"), ("S2", "referència nova"), ("S3", "TTS nou")]:
        t0 = time.time()
        preds, rec = nested_protocol(d, name, configs, a.n_jobs)
        all_rec.append(rec)
        print(f"\n{name} ({what}) [{time.time() - t0:.0f} s]")
        for fam in FAMILIES + ["SELECTED"]:
            lo, hi = cluster_ci(d, y, preds[fam], a.n_boot)
            print(f"  {fam:13s} UAR={balanced_accuracy_score(y, preds[fam]):.3f}  IC95=({lo:.3f}, {hi:.3f})")
        sel = rec[rec.selected]
        print("  Família triada per fold extern:", sel["family"].value_counts().to_dict())
        print("  UAR per fold (SELECTED):", np.round(sel.sort_values("fold")["test_uar"].to_numpy(), 3))
    out = pd.concat(all_rec)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nDetall (hiperparàmetres per fold, inner vs test): {a.out}")


if __name__ == "__main__":
    main()