"""
Avalua la capacitat de predir l'emoció de la veu (angry, happy, neutral, sad) 
a partir dels embeddings WavLM, comparant quatre classificadors 
(LogReg, RandomForest, SVM-RBF i HistGB) mitjançant validació creuada aniuada.

Selecciona la millor capa de WavLM i els hiperparàmetres per a cada classificador.

Avalua la generalització en tres escenaris:

S1 (textos nous),

S2 (referències noves) i

S3 (sistemes TTS nous),

utilitzant la mètrica UAR i intervals de confiança del 95%.

Genera `outputs/e0/wavlm_nested.csv`

Desa els resultats de cada classificador per als protocols S1, S2 i S3.

Inclou la capa WavLM seleccionada per cada fold,

els millors hiperparàmetres,

la UAR de la validació interna i

la UAR del test extern.

Serveix per comparar la capacitat de predicció emocional dels embeddings 
WavLM entre classificadors i protocols.
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

# Arrel del projecte (dos nivells per sobre d'aquest script)
ROOT = Path(__file__).resolve().parents[2]
# SEED: llavor per a la reproductibilitat; INNER_K: nombre màxim de folds de la CV interna
SEED, INNER_K = 42, 5
# Classes de l'etiqueta objectiu (ordre fix per a la matriu de confusió)
LABELS = ["angry", "happy", "neutral", "sad"]
# Columna que defineix els grups de cada protocol (evita que un mateix grup caigui a train i test):
# S1 = text, S2 = àudio de referència, S3 = sistema TTS
GROUP_COL = {"S1": "group_text", "S2": "reference_id", "S3": "tts_system"}
FAMILIES = ["LogReg", "RandomForest", "SVM-RBF", "HistGB"]

# Dimensió alta (768 o 1536) amb ~1000 mostres: més regularització i gamma més petita que a eGeMAPS.
# Graella completa d'hiperparàmetres per família
GRIDS = {
    "LogReg": {"C": [1e-4, 1e-3, 1e-2, 1e-1, 1]},
    "RandomForest": {"max_depth": [4, 8, 12], "min_samples_leaf": [2, 5], "max_features": ["sqrt"]},
    "SVM-RBF": {"C": [0.1, 1, 10, 100], "gamma": [3e-4, 1e-3, 3e-3, 1e-2]},
    "HistGB": {"learning_rate": [0.1], "max_leaf_nodes": [3, 7], "l2_regularization": [0, 1]},
}
# Graella mínima (una sola combinació per família) per fer proves ràpides amb --quick
QUICK = {"LogReg": {"C": [1e-2]}, "RandomForest": {"max_depth": [4], "min_samples_leaf": [5],
         "max_features": ["sqrt"]}, "SVM-RBF": {"C": [1], "gamma": [1e-3]},
         "HistGB": {"learning_rate": [0.1], "max_leaf_nodes": [3], "l2_regularization": [0]}}


def make_model(family: str, p: dict):
    """Construeix un pipeline (estandarditzador + classificador) per a la família i
    els hiperparàmetres donats. class_weight='balanced' compensa el desbalanç de classes."""
    if family == "LogReg":
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", **p)
    elif family == "RandomForest":
        # n_jobs=1: el paral·lelisme es fa a fora (joblib), no dins del model
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=SEED, n_jobs=1, **p)
    elif family == "SVM-RBF":
        clf = SVC(kernel="rbf", class_weight="balanced", **p)
    else:
        # HistGB: nombre fix d'iteracions, sense early stopping (evita usar dades de validació internes)
        clf = HistGradientBoostingClassifier(max_iter=100, early_stopping=False,
                                             class_weight="balanced", random_state=SEED, **p)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])  # scaler només amb train


def expand(grids: dict, layers: list):
    """Configuracions (família, params, capa). La capa entra a la cerca interna."""
    out = []
    for fam in FAMILIES:
        keys = list(grids[fam])
        # Producte cartesià de tots els valors dels hiperparàmetres de la família
        for v in itertools.product(*grids[fam].values()):
            # Cada combinació es repeteix per cada capa candidata
            out += [(fam, dict(zip(keys, v)), layer) for layer in layers]
    return out


def fit_score(family, params, layer, Xtr, y, tr, va) -> float:
    """Entrena una configuració sobre els índexs `tr` i retorna la UAR (balanced accuracy)
    sobre els índexs de validació `va`. Xtr té forma (n, capes, dims)."""
    Xl = Xtr[:, layer]  # selecciona la capa de WavLM -> (n, dims)
    m = make_model(family, params).fit(Xl[tr], y[tr])
    return balanced_accuracy_score(y[va], m.predict(Xl[va]))


def nested_protocol(d: pd.DataFrame, cube: np.ndarray, name: str, configs, n_jobs: int):
    """CV aniuada per a un protocol (S1/S2/S3). El fold extern avalua; el fold intern
    tria (família a família) els millors hiperparàmetres i la millor capa."""
    fold = d[f"fold_{name}"].to_numpy()          # fold extern predefinit al fitxer de splits
    y_all = d["generation_emotion"].to_numpy()   # etiquetes
    g_all = d[GROUP_COL[name]].to_numpy()        # grups del protocol
    # Prediccions externes de cada família (una per clip)
    preds = {f: np.empty(len(d), dtype=object) for f in FAMILIES}
    records = []
    for k in np.unique(fold):
        # Train = tots els folds excepte k; test = fold k
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        Xtr, y, g = cube[tr], y_all[tr], g_all[tr]  # (n_tr, L, D)
        # Mai més folds interns que grups disponibles
        kk = min(INNER_K, len(np.unique(g)))
        # CV interna estratificada per classe i agrupada, només dins del train extern
        splits = list(StratifiedGroupKFold(n_splits=kk, shuffle=True, random_state=SEED)
                      .split(np.zeros(len(tr)), y, g))
        # Totes les parelles (configuració, split intern), avaluades en paral·lel
        jobs = [(ci, si) for ci in range(len(configs)) for si in range(len(splits))]
        res = Parallel(n_jobs=n_jobs)(
            delayed(fit_score)(*configs[ci], Xtr, y, *splits[si]) for ci, si in jobs)
        # Matriu (configuracions x splits interns) amb la UAR de cada combinació
        inner = np.zeros((len(configs), len(splits)))
        for (ci, si), r in zip(jobs, res):
            inner[ci, si] = r
        inner_mean = inner.mean(axis=1)  # UAR interna mitjana per configuració
        for fam in FAMILIES:  # millor (params, capa) de cada família, només amb el train
            cis = [i for i, c in enumerate(configs) if c[0] == fam]
            b = cis[int(np.argmax(inner_mean[cis]))]  # índex de la millor configuració
            f_, p_, layer = configs[b]
            # Reentrena amb tot el train extern i prediu el fold de test
            m = make_model(f_, p_).fit(Xtr[:, layer], y)
            preds[fam][te] = m.predict(cube[te][:, layer])
            # Desa el detall d'aquest fold: configuració triada, UAR interna i UAR de test
            records.append({"protocol": name, "fold": int(k), "family": fam, "layer": layer,
                            "params": str(p_), "inner_uar": inner_mean[b],
                            "test_uar": balanced_accuracy_score(y_all[te], preds[fam][te])})
    return preds, pd.DataFrame(records)


def cluster_ci(d: pd.DataFrame, y, pred, n_boot: int):
    """Interval de confiança al 95% de la UAR amb bootstrap per clústers:
    es remostregen textos (group_text) sencers, no clips individuals, per respectar
    la dependència entre clips que comparteixen text."""
    rng = np.random.default_rng(SEED)
    groups = d["group_text"].unique()
    # Índexs dels clips de cada text
    by_g = {g: np.flatnonzero((d["group_text"] == g).to_numpy()) for g in groups}
    # Cada rèplica: tria textos amb reemplaçament, concatena els seus clips i calcula la UAR
    boots = [balanced_accuracy_score(y[i], pred[i]) for i in
             (np.concatenate([by_g[g] for g in rng.choice(groups, len(groups))])
              for _ in range(n_boot))]
    return np.percentile(boots, [2.5, 97.5])


def main() -> None:
    # ---------------- Arguments de línia d'ordres ----------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=ROOT / "outputs/features/wavlm_base_plus_meanstd_sr16000.npz")
    ap.add_argument("--splits", default=ROOT / "configs/splits/splits_v1.csv")
    ap.add_argument("--out", default=ROOT / "outputs/e0/wavlm_nested.csv")
    # "mean": només la mitjana; "meanstd": mitjana i desviació concatenades (dimensió doble)
    ap.add_argument("--pool", choices=["mean", "meanstd"], default="mean")
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9, 12],
                    help="capes candidates (0 = projecció CNN); es trien a la CV interna")
    ap.add_argument("--n-jobs", type=int, default=-1)       # -1 = tots els nuclis
    ap.add_argument("--n-boot", type=int, default=2000)     # rèpliques del bootstrap
    ap.add_argument("--quick", action="store_true", help="graella reduïda, només per provar")
    ap.add_argument("--expect-n", type=int, default=1248)   # nombre de clips esperat (control)
    a = ap.parse_args()

    # ---------------- Càrrega i alineació de dades ----------------
    z = np.load(a.features, allow_pickle=False)
    ids = z["audio_id"].astype(str)
    # cube: (N clips, L capes, D dims); amb meanstd, s'concatenen mean i std a l'eix de dims
    cube = z["mean"] if a.pool == "mean" else np.concatenate([z["mean"], z["std"]], axis=-1)
    s = pd.read_csv(a.splits, dtype={"audio_id": str, "reference_id": str})
    # Posició de cada audio_id dins del cub d'embeddings
    pos = pd.DataFrame({"audio_id": ids, "row": np.arange(len(ids))})
    # Uneix splits i posicions; validate garanteix correspondència 1 a 1
    d = s.merge(pos, on="audio_id", validate="one_to_one").reset_index(drop=True)
    cube = cube[d["row"].to_numpy()].astype("float64")  # alineat amb l'ordre de d
    # En mode quick, només una capa
    layers = [6] if a.quick else a.layers
    # Comprovacions: nombre de clips, absència de NaN/inf i capes dins del rang
    assert len(d) == a.expect_n and np.isfinite(cube).all()
    assert max(layers) < cube.shape[1], f"el model té {cube.shape[1]} capes (0..{cube.shape[1] - 1})"
    configs = expand(QUICK if a.quick else GRIDS, layers)
    print(f"sklearn {sklearn.__version__} | pooling: {a.pool} | dims: {cube.shape[2]} | capes: {layers}"
          f" | configuracions: {len(configs)} ({'quick' if a.quick else 'graella completa'})"
          f" | CV interna k<={INNER_K}")

    # ---------------- Avaluació per protocol ----------------
    y = d["generation_emotion"].to_numpy()
    all_rec = []
    print("\nCV aniuada (atzar = 0.25). UAR agrupat de les prediccions externes + IC95 cluster bootstrap per text")
    for name, what in [("S1", "textos nous"), ("S2", "referència nova"), ("S3", "TTS nou")]:
        t0 = time.time()
        preds, rec = nested_protocol(d, cube, name, configs, a.n_jobs)
        all_rec.append(rec)
        print(f"\n{name} ({what}) [{time.time() - t0:.0f} s]")
        for fam in FAMILIES:
            # IC95 de la UAR global de la família
            lo, hi = cluster_ci(d, y, preds[fam], a.n_boot)
            r = rec[rec.family == fam].sort_values("fold")
            print(f"  {fam:13s} UAR={balanced_accuracy_score(y, preds[fam]):.3f}  IC95=({lo:.3f}, {hi:.3f})"
                  f"  per fold={np.round(r['test_uar'].to_numpy(), 3)}"
                  f"  capa triada={r['layer'].tolist()}")
        # Matriu de confusió només per a LogReg (com a referència)
        cm = confusion_matrix(y, preds["LogReg"], labels=LABELS)
        print("  Matriu de confusió LogReg (files = real, columnes = predit):")
        print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())
    # ---------------- Desa el detall de tots els folds ----------------
    out = pd.concat(all_rec)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nDetall (hiperparàmetres i capa per fold, inner vs test): {a.out}")


if __name__ == "__main__":
    main()