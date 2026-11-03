"""
PAS 2 de 2 — Avaluació zero-shot: UAR, IC95 i matrius de confusió.
 
Llegeix els features/<model>.csv del pas 1 i els compara amb l'etiqueta real de cada text únic.
La predicció és la classe amb més probabilitat (argmax); no s'executa ni s'entrena cap model.
Calcula per model la UAR (mitjana dels recalls de les 4 classes), el seu IC95 per bootstrap
sobre textos únics, accuracy, macro-F1 i recall per classe.
 
Genera a outputs/text_baseline/:
  metrics_zero_shot.csv            Taula de resultats per model, ordenada per UAR.
  confusion_zero_shot_<model>.csv  Matriu de confusió de cada model (files = real, columnes = predit).
  predictions_zero_shot.csv        Predicció text a text, per analitzar errors.
 
Dependències: outputs generats per text_emotion_extraction.py (features/*.csv, que
s'uneixen per la columna `text`) i data/processed/emis_manifest.csv (text, text_emotion);
numpy, pandas, scikit-learn. No cal GPU ni internet.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

# Arrel del projecte: l'script és a src/emis_research/, per tant pugem dos nivells.
ROOT = Path(__file__).resolve().parents[2]
SEED = 42                                        # llavor fixa perquè el bootstrap sigui reproduïble
LABELS = ["angry", "happy", "neutral", "sad"]    # ordre fix de les 4 classes (el mateix per a tot: columnes, matrius...)


def uar(y, p):
    """UAR (Unweighted Average Recall): mitjana dels recalls de cada classe, sense ponderar pel suport.

    El recall d'una classe és la fracció dels textos d'aquella classe que el model ha encertat.
    Així, una classe amb pocs textos pesa igual que una amb molts, cosa que fa la mètrica
    robusta davant de classes desequilibrades. Amb classes equilibrades coincideix amb l'accuracy.

    y: etiquetes reals (array); p: etiquetes predites (array de la mateixa mida).
    """
    # Per a cada classe present a y: es filtren les prediccions dels textos d'aquella classe
    # i es calcula quina proporció coincideix amb la classe. Després, mitjana de classes.
    return float(np.mean([(p[y == l] == l).mean() for l in LABELS if (y == l).any()]))


def boot_ci(y, p, n_boot):
    """Interval de confiança del 95% del UAR, calculat per bootstrap sobre text únic.

    Procediment:
      1. Es pren una mostra aleatòria AMB reemplaçament de N textos (N = nombre total de textos).
      2. Se'n recalcula el UAR.
      3. Es repeteix n_boot vegades.
    L'interval és el percentil 2.5 i el 97.5 d'aquests n_boot valors d'UAR.

    Amb pocs textos l'interval és ample: això reflecteix la incertesa real de la mesura.
    """
    rng = np.random.default_rng(SEED)
    N = len(y)
    # rng.integers(0, N, N) dona N índexs aleatoris (amb repetició) = una mostra bootstrap.
    # Es genera un vector d'índexs per iteració i s'hi calcula el UAR.
    v = [uar(y[i], p[i]) for i in (rng.integers(0, N, N) for _ in range(n_boot))]
    return np.percentile(v, [2.5, 97.5])


def main() -> None:
    # ---------- Arguments de línia d'ordres ----------
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=ROOT / "data/processed/emis_manifest.csv")      # àudios + text + etiqueta
    ap.add_argument("--feat-dir", default=ROOT / "outputs/text_baseline/features")        # sortides dels models (un CSV per model)
    ap.add_argument("--out-dir", default=ROOT / "outputs/text_baseline")                  # on es desen els resultats
    ap.add_argument("--n-boot", type=int, default=2000)                                   # nombre de remostrejos bootstrap
    ap.add_argument("--models", nargs="+", default=None,
                    help="noms de fitxer (sense .csv) a avaluar; per defecte, tots els de --feat-dir")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)   # crea la carpeta de sortida si no existeix

    # ---------- PAS 1: textos únics + etiqueta real ----------
    # Ha de ser el mateix conjunt (i el mateix ordre) que es va usar a l'extracció de features.
    # Unitats = textos únics (frase literal). `text_emotion` només s'usa per avaluar.
    man = pd.read_csv(a.manifest, dtype=str, usecols=["text", "text_emotion"])
    # Comprovació de coherència: una mateixa frase no pot tenir dues etiquetes diferents.
    assert man.groupby("text")["text_emotion"].nunique().max() == 1, "mateix text amb etiquetes diferents"
    # Diversos àudios poden compartir text: ens quedem amb una sola fila per frase.
    U = man.drop_duplicates("text").reset_index(drop=True)
    # Comprovació: hi ha d'haver exactament les 4 classes esperades (ni més ni menys).
    assert set(U.text_emotion) == set(LABELS), "falten classes al conjunt de textos"
    y = U.text_emotion.to_numpy()   # vector d'etiquetes reals
    print(f"{len(man)} àudios -> {len(U)} textos únics | classes: {U.text_emotion.value_counts().to_dict()}")

    # Fitxers de features a avaluar: tots els CSV de la carpeta, o només els models indicats amb --models.
    files = [f for f in sorted(Path(a.feat_dir).glob("*.csv")) if not a.models or f.stem in a.models]
    assert files, f"no hi ha features a {a.feat_dir}"

    # ---------- PAS 2: per a cada model, predir i calcular mètriques ----------
    rows, preds = [], []   # rows: una fila de mètriques per model; preds: prediccions text a text
    for f in files:
        name = f.stem   # el nom del model és el nom del fitxer sense extensió
        # Les probabilitats s'uneixen per la frase literal (`text`), no per posició, perquè així
        # l'ordre del CSV del model no importa. validate="one_to_one" garanteix 1 fila per text.
        F = U[["text"]].merge(pd.read_csv(f), on="text", how="left", validate="one_to_one")
        pc = [f"p_{e}" for e in LABELS]   # noms de columnes: p_angry, p_happy, p_neutral, p_sad
        # Si algun text no ha rebut sortida del model, hi haurà NaN: ho detectem i aturem.
        assert F[pc].notna().all().all(), f"{f.name}: textos sense sortida"
        p = F[pc].to_numpy()   # matriu (n_textos x 4) de probabilitats
        # Predicció zero-shot: la classe amb més probabilitat per a cada text. Cap entrenament.
        pred = np.array(LABELS)[p.argmax(axis=1)]
        lo, hi = boot_ci(y, pred, a.n_boot)   # IC95 del UAR
        rec = recall_score(y, pred, labels=LABELS, average=None, zero_division=0)  # un recall per classe
        # Mètriques globals del model.
        r = {"model": name, "n_texts": len(y), "uar": uar(y, pred), "uar_ci_lo": lo, "uar_ci_hi": hi,
             "accuracy": accuracy_score(y, pred),
             "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)}
        # Mètriques per classe: recall i suport.
        for l, rc in zip(LABELS, rec):
            r[f"recall_{l}"] = rc
            r[f"support_{l}"] = int((y == l).sum())      # nombre de textos reals de cada classe
        rows.append(r)
        # Matriu de confusió: files = classe real, columnes = classe predita.
        # La diagonal són els encerts; la resta mostra quines emocions es confonen entre elles.
        pd.DataFrame(confusion_matrix(y, pred, labels=LABELS), index=[f"true_{l}" for l in LABELS],
                     columns=[f"pred_{l}" for l in LABELS]).to_csv(out / f"confusion_zero_shot_{name}.csv")
        # Predicció per text (per poder analitzar els errors un a un): afegeix al DataFrame de textos
        # el nom del model, la predicció, si és correcta i les 4 probabilitats.
        preds.append(U.assign(model=name, pred=pred, correct=(pred == y),
                              **{c: p[:, i] for i, c in enumerate(pc)}))

    # ---------- PAS 3: desar i mostrar la taula, ordenada per UAR ----------
    M = pd.DataFrame(rows).sort_values("uar", ascending=False)   # millor model primer
    M.to_csv(out / "metrics_zero_shot.csv", index=False)
    pd.concat(preds).to_csv(out / "predictions_zero_shot.csv", index=False)

    # Taula resumida per mostrar per pantalla (valors formatats amb 3 decimals).
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