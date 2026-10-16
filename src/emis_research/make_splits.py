# src/emis_research/make_splits.py
"""Splits agrupats reproduibles per a EMIS: S1 (text), S2 (referència), S3 (TTS)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "processed" / "emis_manifest.csv"
OUT_DIR = ROOT / "configs" / "splits"
VERSION, SEED, N_FOLDS_TEXT = "v1", 42, 5


def load_manifest(path: Path = MANIFEST) -> pd.DataFrame:
    m = pd.read_csv(path, dtype=str)  # str: conserva zeros a l'esquerra
    m["group_text"] = m["text_category"] + "_" + m["text_id"]
    return m


def split_text(m: pd.DataFrame) -> np.ndarray:
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS_TEXT, shuffle=True, random_state=SEED)
    fold = np.full(len(m), -1)
    for k, (_, test_idx) in enumerate(
        sgkf.split(m, y=m["text_category"], groups=m["group_text"])
    ):
        fold[test_idx] = k
    return fold


def split_by_column(m: pd.DataFrame, col: str) -> np.ndarray:
    levels = {v: i for i, v in enumerate(sorted(m[col].unique()))}
    return m[col].map(levels).to_numpy()


def build_splits(m: pd.DataFrame) -> pd.DataFrame:
    s = m[["audio_id", "group_text", "reference_id", "tts_system",
           "generation_emotion"]].copy()
    s["fold_S1"] = split_text(m)
    s["fold_S2"] = split_by_column(m, "reference_id")
    s["fold_S3"] = split_by_column(m, "tts_system")
    return s


GROUP_COL = {"S1": "group_text", "S2": "reference_id", "S3": "tts_system"}


def validate(s: pd.DataFrame) -> None:
    assert s["audio_id"].is_unique and len(s) == 1248
    assert s["group_text"].nunique() == 104
    for name, gcol in GROUP_COL.items():
        f = s[f"fold_{name}"]
        assert (f >= 0).all(), f"{name}: mostres sense fold"
        for k in sorted(f.unique()):
            test_g = set(s.loc[f == k, gcol])
            train_g = set(s.loc[f != k, gcol])
            assert not (test_g & train_g), f"{name} fold {k}: solapament de {gcol}"
    # S1: equilibri per emoció i TTS dins de cada fold (per construcció)
    for k, part in s.groupby("fold_S1"):
        assert part["generation_emotion"].value_counts().nunique() == 1, f"S1 fold {k}"
        assert part["tts_system"].value_counts().nunique() == 1, f"S1 fold {k}"


def main() -> None:
    m = load_manifest()
    s = build_splits(m)
    validate(s)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT_DIR / f"splits_{VERSION}.csv", index=False)
    meta = {
        "version": VERSION, "seed": SEED, "n_folds_text": N_FOLDS_TEXT,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "sklearn": sklearn.__version__, "pandas": pd.__version__,
        "fold_sizes": {n: s[f"fold_{n}"].value_counts().sort_index().to_dict()
                       for n in GROUP_COL},
    }
    (OUT_DIR / f"splits_{VERSION}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("splits_v1.csv sha256:",
          hashlib.sha256((OUT_DIR / f"splits_{VERSION}.csv").read_bytes()).hexdigest())


if __name__ == "__main__":
    main()