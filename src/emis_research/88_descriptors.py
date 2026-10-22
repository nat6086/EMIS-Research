"""Predicció de generation_emotion amb els 88 descriptors eGeMAPSv02 (24 kHz natiu).

Mateixa estructura que qc_shortcuts.py: CV aniuada, 4 famílies de classificadors, splits
S1/S2/S3. La CV interna només veu el train de cada fold extern (agrupada segons el
protocol); el fold de test s'usa una sola vegada. No hi ha selecció entre famílies:
es reporta cada una. Només lectura: no modifica dades ni splits.
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
SEED, INNER_K = 42, 5
LABELS = ["angry", "happy", "neutral", "sad"]
GROUP_COL = {"S1": "group_text", "S2": "reference_id", "S3": "tts_system"}
FAMILIES = ["LogReg", "RandomForest", "SVM-RBF", "HistGB"]

GRIDS = {
    "LogReg": {"C": [0.01, 0.1, 1, 10, 100]},
    "RandomForest": {"max_depth": [4, 8, 12], "min_samples_leaf": [2, 5, 10],
                     "max_features": ["sqrt", 1.0]},
    "SVM-RBF": {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]},
    "HistGB": {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [3, 7, 15],
               "l2_regularization": [0, 1]},
}
QUICK = {"LogReg": {"C": [0.1, 1]}, "RandomForest": {"max_depth": [4], "min_samples_leaf": [5],
         "max_features": ["sqrt"]}, "SVM-RBF": {"C": [1], "gamma": [0.01]},
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


def expand(grids: dict):
    out = []
    for fam in FAMILIES:
        keys = list(grids[fam])
        out += [(fam, dict(zip(keys, v))) for v in itertools.product(*grids[fam].values())]
    return out


def fit_score(family, params, X, y, tr, va) -> float:
    m = make_model(family, params).fit(X[tr], y[tr])
    return balanced_accuracy_score(y[va], m.predict(X[va]))


def nested_protocol(d: pd.DataFrame, feats: list, name: str, configs, n_jobs: int):
    fold = d[f"fold_{name}"].to_numpy()
    y_all = d["generation_emotion"].to_numpy()
    X_all = d[feats].to_numpy(dtype="float64")
    g_all = d[GROUP_COL[name]].to_numpy()
    preds = {f: np.empty(len(d), dtype=object) for f in FAMILIES}
    records = []
    for k in np.unique(fold):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        X, y, g = X_all[tr], y_all[tr], g_all[tr]
        kk = min(INNER_K, len(np.unique(g)))
        splits = list(StratifiedGroupKFold(n_splits=kk, shuffle=True, random_state=SEED)
                      .split(X, y, g))
        jobs = [(ci, si) for ci in range(len(configs)) for si in range(len(splits))]
        res = Parallel(n_jobs=n_jobs)(
            delayed(fit_score)(configs[ci][0], configs[ci][1], X, y, *splits[si])
            for ci, si in jobs)
        inner = np.zeros((len(configs), len(splits)))
        for (ci, si), r in zip(jobs, res):
            inner[ci, si] = r
        inner_mean = inner.mean(axis=1)
        for fam in FAMILIES:  # millor configuració de cada família, només amb el train
            cis = [i for i, c in enumerate(configs) if c[0] == fam]
            b = cis[int(np.argmax(inner_mean[cis]))]
            m = make_model(*configs[b]).fit(X, y)
            preds[fam][te] = m.predict(X_all[te])
            records.append({"protocol": name, "fold": int(k), "family": fam,
                            "params": str(configs[b][1]), "inner_uar": inner_mean[b],
                            "test_uar": balanced_accuracy_score(y_all[te], preds[fam][te])})
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
    ap.add_argument("--features", default=ROOT / "outputs/features/egemaps_v02_functionals_sr24000.csv")
    ap.add_argument("--splits", default=ROOT / "configs/splits/splits_v1.csv")
    ap.add_argument("--out", default=ROOT / "outputs/e0/ege88_nested.csv")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="graella reduïda, només per provar")
    ap.add_argument("--expect-n", type=int, default=1248)
    a = ap.parse_args()
    ft = pd.read_csv(a.features, dtype={"audio_id": str})
    s = pd.read_csv(a.splits, dtype={"audio_id": str, "reference_id": str})
    feats = [c for c in ft.columns if c != "audio_id"]
    d = s.merge(ft, on="audio_id", validate="one_to_one")
    assert len(d) == a.expect_n and len(feats) == 88 and d[feats].notna().all().all()
    configs = expand(QUICK if a.quick else GRIDS)
    print(f"sklearn {sklearn.__version__} | descriptors: {len(feats)} | configuracions: {len(configs)} "
          f"({'quick' if a.quick else 'graella completa'}) | CV interna k<={INNER_K}")

    y = d["generation_emotion"].to_numpy()
    all_rec = []
    print("\nCV aniuada (atzar = 0.25). UAR agrupat de les prediccions externes + IC95 cluster bootstrap per text")
    for name, what in [("S1", "textos nous"), ("S2", "referència nova"), ("S3", "TTS nou")]:
        t0 = time.time()
        preds, rec = nested_protocol(d, feats, name, configs, a.n_jobs)
        all_rec.append(rec)
        print(f"\n{name} ({what}) [{time.time() - t0:.0f} s]")
        for fam in FAMILIES:
            lo, hi = cluster_ci(d, y, preds[fam], a.n_boot)
            v = rec[rec.family == fam].sort_values("fold")["test_uar"].to_numpy()
            print(f"  {fam:13s} UAR={balanced_accuracy_score(y, preds[fam]):.3f}  IC95=({lo:.3f}, {hi:.3f})"
                  f"  per fold={np.round(v, 3)}")
        cm = confusion_matrix(y, preds["LogReg"], labels=LABELS)
        print("  Matriu de confusió LogReg (files = real, columnes = predit):")
        print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())
    out = pd.concat(all_rec)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nDetall (hiperparàmetres per fold, inner vs test): {a.out}")


if __name__ == "__main__":
    main()