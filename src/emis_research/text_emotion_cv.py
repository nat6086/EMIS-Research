"""Evaluació ZERO-SHOT de la branca textual d'EMIS (4 classes: angry, happy, neutral, sad).

Per cada model preentrenat: pred = argmax(p_angry, p_happy, p_neutral, p_sad). No s'entrena res amb EMIS.
Unitat d'anàlisi: text únic (cada frase un cop). Target: `text_emotion`.
Les etiquetes només s'usen aquí, per calcular mètriques; mai com a input dels models.

Mètriques per model: UAR (principal) amb IC95 per bootstrap sobre text únic, Accuracy, Macro-F1,
recall i suport per classe, matriu de confusió i prediccions per text.

Per executar:
python src/emis_research/text_emotion_zero_shot.py
python src/emis_research/text_emotion_zero_shot.py --models qwen3_4b qwen3_8b deberta_zeroshot
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
LABELS = ["angry", "happy", "neutral", "sad"]


def uar(y, p):
    return float(np.mean([(p[y == l] == l).mean() for l in LABELS if (y == l).any()]))


def boot_ci(y, p, n_boot):
    rng = np.random.default_rng(SEED)
    N = len(y)
    v = [uar(y[i], p[i]) for i in (rng.integers(0, N, N) for _ in range(n_boot))]
    return np.percentile(v, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=ROOT / "data/processed/emis_manifest.csv")
    ap.add_argument("--feat-dir", default=ROOT / "outputs/text_baseline/features")
    ap.add_argument("--out-dir", default=ROOT / "outputs/text_baseline")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--models", nargs="+", default=None,
                    help="noms de fitxer (sense .csv) a avaluar; per defecte, tots els de --feat-dir")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Unitats = textos únics (frase literal). text_emotion només per avaluar.
    man = pd.read_csv(a.manifest, dtype=str, usecols=["text", "text_emotion"])
    assert man.groupby("text")["text_emotion"].nunique().max() == 1, "mateix text amb etiquetes diferents"
    U = man.drop_duplicates("text").reset_index(drop=True)
    assert set(U.text_emotion) == set(LABELS), "falten classes al conjunt de textos"
    y = U.text_emotion.to_numpy()
    print(f"{len(man)} àudios -> {len(U)} textos únics | classes: {U.text_emotion.value_counts().to_dict()}")

    files = [f for f in sorted(Path(a.feat_dir).glob("*.csv")) if not a.models or f.stem in a.models]
    assert files, f"no hi ha features a {a.feat_dir}"

    rows, preds = [], []
    for f in files:
        name = f.stem
        F = U[["text"]].merge(pd.read_csv(f), on="text", how="left", validate="one_to_one")
        pc = [f"p_{e}" for e in LABELS]
        assert F[pc].notna().all().all(), f"{f.name}: textos sense sortida"
        p = F[pc].to_numpy()
        pred = np.array(LABELS)[p.argmax(axis=1)]
        lo, hi = boot_ci(y, pred, a.n_boot)
        rec = recall_score(y, pred, labels=LABELS, average=None, zero_division=0)
        r = {"model": name, "n_texts": len(y), "uar": uar(y, pred), "uar_ci_lo": lo, "uar_ci_hi": hi,
             "accuracy": accuracy_score(y, pred),
             "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)}
        for l, rc in zip(LABELS, rec):
            r[f"recall_{l}"] = rc
            r[f"support_{l}"] = int((y == l).sum())
        rows.append(r)
        pd.DataFrame(confusion_matrix(y, pred, labels=LABELS), index=[f"true_{l}" for l in LABELS],
                     columns=[f"pred_{l}" for l in LABELS]).to_csv(out / f"confusion_zero_shot_{name}.csv")
        preds.append(U.assign(model=name, pred=pred, correct=(pred == y),
                              **{c: p[:, i] for i, c in enumerate(pc)}))

    M = pd.DataFrame(rows).sort_values("uar", ascending=False)
    M.to_csv(out / "metrics_zero_shot.csv", index=False)
    pd.concat(preds).to_csv(out / "predictions_zero_shot.csv", index=False)

    T = pd.DataFrame({"Model": M.model, "UAR (IC95)": [f"{r.uar:.3f} ({r.uar_ci_lo:.3f}-{r.uar_ci_hi:.3f})"
                                                       for r in M.itertuples()],
                      "Macro-F1": M.macro_f1.map("{:.3f}".format), "Acc": M.accuracy.map("{:.3f}".format),
                      **{f"R_{l}": M[f"recall_{l}"].map("{:.3f}".format) for l in LABELS}})
    pd.set_option("display.width", 200)
    print("\n=== ZERO-SHOT (4 classes; ordenat per UAR) ===")
    print(T.to_string(index=False))
    print(f"\nResultats: {out}")


if __name__ == "__main__":
    main()