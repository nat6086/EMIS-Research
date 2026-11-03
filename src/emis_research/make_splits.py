# src/emis_research/make_splits.py
"""
Genera els splits reproduïbles del dataset EMIS per a tres protocols d'avaluació:

S1 (textos nous, 5 folds),

S2 (referències noves) i

S3 (sistemes TTS nous).

Comprova que no hi hagi solapament de grups entre train i test, evitant fugues d'informació.

Genera configs/splits/splits_v1.csv

Desa l'assignació de cada àudio als folds dels tres protocols (S1, S2 i S3).

S'utilitza per entrenar i avaluar els classificadors sota diferents escenaris de generalització.

Genera configs/splits/splits_v1.json

Desa les metadades dels splits: versió, llavor aleatòria, versions de les llibreries, 
distribució dels folds i hash del manifest.

Serveix per garantir la traçabilitat i reproductibilitat dels splits.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

# Arrel del projecte (dos nivells per sobre d'aquest fitxer)
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "processed" / "emis_manifest.csv"  # llista d'àudios amb metadades
OUT_DIR = ROOT / "configs" / "splits"                          # carpeta de sortida dels splits
# VERSION: etiqueta dels fitxers de sortida; SEED: llavor per a la reproductibilitat;
# N_FOLDS_TEXT: nombre de folds del split per textos (S1)
VERSION, SEED, N_FOLDS_TEXT = "v1", 42, 5


def load_manifest(path: Path = MANIFEST) -> pd.DataFrame:
    """Carrega el manifest i crea la columna de grup de text (categoria + identificador)."""
    m = pd.read_csv(path, dtype=str)  # str: conserva zeros a l'esquerra
    # el mateix text_id pot repetir-se entre categories, per això es combina amb la categoria
    m["group_text"] = m["text_category"] + "_" + m["text_id"]
    return m


def split_text(m: pd.DataFrame) -> np.ndarray:
    """S1 (textos nous): 5 folds on tots els clips d'un mateix text van al mateix fold,
    estratificats per categoria de text."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS_TEXT, shuffle=True, random_state=SEED)
    fold = np.full(len(m), -1)  # -1 = sense assignar (la validació ho detectaria)
    for k, (_, test_idx) in enumerate(
        sgkf.split(m, y=m["text_category"], groups=m["group_text"])
    ):
        fold[test_idx] = k  # els clips de test d'aquesta iteració formen el fold k
    return fold


def split_by_column(m: pd.DataFrame, col: str) -> np.ndarray:
    """S2 i S3: leave-one-group-out. Cada valor de la columna (referència o TTS)
    forma el seu propi fold."""
    # assignem un número de fold a cada valor, en ordre alfabètic (determinista)
    levels = {v: i for i, v in enumerate(sorted(m[col].unique()))}
    return m[col].map(levels).to_numpy()


def build_splits(m: pd.DataFrame) -> pd.DataFrame:
    """Construeix la taula de splits: metadades de cada àudio + fold dels tres protocols."""
    s = m[["audio_id", "group_text", "reference_id", "tts_system",
           "generation_emotion"]].copy()
    s["fold_S1"] = split_text(m)                      # textos nous
    s["fold_S2"] = split_by_column(m, "reference_id")  # referències noves
    s["fold_S3"] = split_by_column(m, "tts_system")    # sistemes TTS nous
    return s


# Columna que defineix els grups de cada protocol
GROUP_COL = {"S1": "group_text", "S2": "reference_id", "S3": "tts_system"}


def validate(s: pd.DataFrame) -> None:
    """Comprova que els splits són correctes; si no, atura l'execució amb un error."""
    # nombre total de clips i identificadors únics; 104 textos diferents
    assert s["audio_id"].is_unique and len(s) == 1248
    assert s["group_text"].nunique() == 104
    for name, gcol in GROUP_COL.items():
        f = s[f"fold_{name}"]
        assert (f >= 0).all(), f"{name}: mostres sense fold"
        for k in sorted(f.unique()):
            test_g = set(s.loc[f == k, gcol])
            train_g = set(s.loc[f != k, gcol])
            # cap grup (text, referència o TTS) pot ser alhora al train i al test:
            # així s'evita la fuita d'informació
            assert not (test_g & train_g), f"{name} fold {k}: solapament de {gcol}"
    # S1: equilibri per emoció i TTS dins de cada fold (per construcció)
    # (si tots els recomptes són iguals, value_counts().nunique() val 1)
    for k, part in s.groupby("fold_S1"):
        assert part["generation_emotion"].value_counts().nunique() == 1, f"S1 fold {k}"
        assert part["tts_system"].value_counts().nunique() == 1, f"S1 fold {k}"


def main() -> None:
    m = load_manifest()
    s = build_splits(m)
    validate(s)  # només es desa si passa totes les comprovacions
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT_DIR / f"splits_{VERSION}.csv", index=False)
    # metadades per a la traçabilitat: versió, llavor, hash del manifest,
    # versions de les llibreries i mida de cada fold
    meta = {
        "version": VERSION, "seed": SEED, "n_folds_text": N_FOLDS_TEXT,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "sklearn": sklearn.__version__, "pandas": pd.__version__,
        "fold_sizes": {n: s[f"fold_{n}"].value_counts().sort_index().to_dict()
                       for n in GROUP_COL},
    }
    (OUT_DIR / f"splits_{VERSION}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    # hash del CSV de splits, per verificar que no s'ha modificat
    print("splits_v1.csv sha256:",
          hashlib.sha256((OUT_DIR / f"splits_{VERSION}.csv").read_bytes()).hexdigest())


if __name__ == "__main__":
    main()