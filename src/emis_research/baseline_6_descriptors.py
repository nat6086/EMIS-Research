"""Baseline trivial amb descriptors de senyal (sense prosòdia fina) + control dins de text.

Objectiu: quantificar quanta part de generation_emotion s'explica només amb durada,
nivell i silencis, sota els splits S1/S2/S3. És un llindar de referència, no un model.
Només lectura: no modifica dades ni splits. Cap hiperparàmetre es tria amb el test.
quant dura, quina energia té, quant silenci conté i quin és el seu pic màxim.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATS = ["duration_s", "rms_dbfs", "low_energy_frac", "lead_sil_s", "trail_sil_s", "peak"]
SEED, N_BOOT = 42, 2000


def within_text_eta2(d: pd.DataFrame, col: str) -> float:
    """eta² de l'emoció després de restar la mitjana de cada text (dins d'un TTS)."""
    r = d[col] - d.groupby("group_text")[col].transform("mean")
    g = r.groupby(d["generation_emotion"])
    ss_b = (g.count() * (g.mean() - r.mean()) ** 2).sum()
    ss_t = ((r - r.mean()) ** 2).sum()
    return float(ss_b / ss_t) if ss_t > 0 else float("nan")


def run_protocol(d: pd.DataFrame, name: str) -> dict:
    fold = d[f"fold_{name}"].to_numpy()
    y = d["generation_emotion"].to_numpy()
    pred = np.empty(len(d), dtype=object)
    per_fold = []
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        sc = StandardScaler().fit(d.loc[tr, FEATS])           # només amb train
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")  # C=1 per defecte
        clf.fit(sc.transform(d.loc[tr, FEATS]), y[tr])
        pred[te] = clf.predict(sc.transform(d.loc[te, FEATS]))
        per_fold.append(balanced_accuracy_score(y[te], pred[te]))
    uar = balanced_accuracy_score(y, pred)
    rng = np.random.default_rng(SEED)                          # cluster bootstrap per text
    groups = d["group_text"].unique()
    by_g = {g: np.flatnonzero((d["group_text"] == g).to_numpy()) for g in groups}
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([by_g[g] for g in rng.choice(groups, len(groups))])
        boots.append(balanced_accuracy_score(y[idx], pred[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"uar_pooled": uar, "ci95": (lo, hi), "per_fold": np.round(per_fold, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", default=ROOT / "outputs/qc/audio_qc.csv")
    ap.add_argument("--splits", default=ROOT / "configs/splits/splits_v1.csv")
    a = ap.parse_args()
    q = pd.read_csv(a.qc, dtype={"audio_id": str, "reference_id": str})
    s = pd.read_csv(a.splits, dtype={"audio_id": str, "reference_id": str})
    need = ["audio_id", "tts_system", "generation_emotion", "flag_long_edge_sil"] + FEATS
    d = q[need].merge(s[["audio_id", "group_text", "fold_S1", "fold_S2", "fold_S3"]],
                      on="audio_id", validate="one_to_one")
    assert len(d) == 1248 and d[FEATS].notna().all().all()

    print("Silenci de vora: fracció amb bandera i mitjana lead_sil_s per TTS x emoció")
    print(d.groupby(["tts_system", "generation_emotion"])["flag_long_edge_sil"].mean().round(3).unstack().to_string())
    print(d.groupby(["tts_system", "generation_emotion"])["lead_sil_s"].mean().round(3).unstack().to_string())

    print("\neta² de l'emoció DINS de text (sense variància entre textos), per TTS:")
    for col in ["duration_s", "rms_dbfs", "lead_sil_s", "low_energy_frac"]:
        print(f"  {col:16s}", {t: round(within_text_eta2(g, col), 3) for t, g in d.groupby("tts_system")})

    print(f"\nBaseline logístic amb {FEATS} (atzar = 0.25). UAR agrupat + IC95 cluster bootstrap per text:")
    for name, what in [("S1", "textos nous"), ("S2", "referència nova"), ("S3", "TTS nou")]:
        r = run_protocol(d, name)
        print(f"  {name} ({what}): UAR={r['uar_pooled']:.3f}  IC95=({r['ci95'][0]:.3f}, {r['ci95'][1]:.3f})  per fold={r['per_fold']}")


if __name__ == "__main__":
    main()