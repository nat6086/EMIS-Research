
"""Auditoria de referències acústiques EMIS i viabilitat del split S2."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "processed" / "emis_manifest.csv"
SPLITS = ROOT / "configs" / "splits" / "splits_v1.csv"
OUT_DIR = ROOT / "reports" / "reference_audit"

REQUIRED = [
    "audio_id",
    "reference_id",
    "generation_emotion",
    "tts_system",
]

SPEAKER_CANDIDATES = [
    "reference_speaker_id",
    "speaker_id",
    "reference_speaker",
    "speaker",
    "speaker_name",
]


def save_table(table: pd.DataFrame, filename: str) -> None:
    path = OUT_DIR / filename
    table.to_csv(path, encoding="utf-8-sig")
    print(f"Desat: {path.relative_to(ROOT)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(MANIFEST, dtype=str)

    missing = [c for c in REQUIRED if c not in m.columns]
    if missing:
        raise ValueError(f"Falten columnes obligatòries: {missing}")

    if m["audio_id"].isna().any() or m["audio_id"].duplicated().any():
        raise ValueError("audio_id conté nuls o duplicats.")

    print("\n" + "=" * 65)
    print("AUDITORIA DE REFERÈNCIES ACÚSTIQUES — EMIS")
    print("=" * 65)

    print(f"\nNombre total d'àudios: {len(m)}")
    print(f"Referències diferents: {m['reference_id'].nunique()}")
    print(f"Emocions: {sorted(m['generation_emotion'].dropna().unique())}")
    print(f"Sistemes TTS: {sorted(m['tts_system'].dropna().unique())}")

    # 1. Valors absents
    print("\n--- 1. VALORS ABSENTS ---")
    print(m[REQUIRED].isna().sum().to_string())

    if m[REQUIRED].isna().any().any():
        raise ValueError(
            "Hi ha valors absents a les columnes obligatòries. "
            "Cal revisar-los abans d'interpretar S2."
        )

    # 2. Referències per emoció
    print("\n--- 2. REFERÈNCIES × EMOCIÓ ---")

    ref_emotion = pd.crosstab(
        m["reference_id"],
        m["generation_emotion"],
    )
    print(ref_emotion.to_string())
    save_table(ref_emotion, "references_by_emotion.csv")

    print("\nNombre de referències diferents per emoció:")
    print(
        m.groupby("generation_emotion")["reference_id"]
        .nunique()
        .to_string()
    )

    # 3. Referències per TTS
    print("\n--- 3. REFERÈNCIES × TTS ---")

    ref_tts = pd.crosstab(
        m["reference_id"],
        m["tts_system"],
    )
    print(ref_tts.to_string())
    save_table(ref_tts, "references_by_tts.csv")

    # 4. Referències per parlant
    print("\n--- 4. REFERÈNCIES × PARLANT ---")

    speaker_col = next(
        (
            c for c in SPEAKER_CANDIDATES
            if c in m.columns and m[c].notna().any()
        ),
        None,
    )

    if speaker_col:
        print(f"Columna detectada: {speaker_col}")
        print(f"Parlants diferents: {m[speaker_col].nunique()}")

        ref_speaker = pd.crosstab(
            m["reference_id"],
            m[speaker_col].fillna("UNKNOWN"),
        )
        print(ref_speaker.to_string())
        save_table(ref_speaker, "references_by_speaker.csv")
    else:
        print(
            "No s'ha identificat una columna de parlant. "
            "No es pot verificar la independència entre speakers."
        )

    # 5. Resum per referència
    print("\n--- 5. RESUM PER REFERÈNCIA ---")

    summary = m.groupby("reference_id").agg(
        n_audios=("audio_id", "count"),
        n_emotions=("generation_emotion", "nunique"),
        n_tts=("tts_system", "nunique"),
    )

    if speaker_col:
        summary["n_speakers"] = (
            m.groupby("reference_id")[speaker_col].nunique()
        )

    print(summary.to_string())
    save_table(summary, "references_summary.csv")

    # 6. Diagnòstic científic
    print("\n--- 6. DIAGNÒSTIC DE S2 ---")

    for ref_id in sorted(m["reference_id"].unique()):
        train = m[m["reference_id"] != ref_id]
        test = m[m["reference_id"] == ref_id]

        train_emotions = set(train["generation_emotion"])
        test_emotions = set(test["generation_emotion"])

        missing_train = test_emotions - train_emotions

        print(f"\nReferència exclosa: {ref_id}")
        print(f"  Train: {len(train)} àudios")
        print(f"  Test:  {len(test)} àudios")
        print(f"  Emocions train: {sorted(train_emotions)}")
        print(f"  Emocions test:  {sorted(test_emotions)}")

        if missing_train:
            print(
                "  ALERTA: emocions presents al test "
                f"però absents del train: {sorted(missing_train)}"
            )

        if len(test_emotions) < m["generation_emotion"].nunique():
            print(
                "  AVÍS: el test no conté totes les emocions. "
                "El Macro F1 per fold pot no ser comparable "
                "sense una política explícita de classes."
            )

    # 7. Comprovar els folds S2 generats
    print("\n--- 7. VERIFICACIÓ DE splits_v1.csv ---")

    if SPLITS.exists():
        s = pd.read_csv(SPLITS, dtype=str)

        needed = {"audio_id", "reference_id", "fold_S2"}
        if not needed.issubset(s.columns):
            raise ValueError(
                f"Falten columnes als splits: {needed - set(s.columns)}"
            )

        if s["audio_id"].duplicated().any():
            raise ValueError("audio_id duplicat al fitxer de splits.")

        merged = m[["audio_id", "reference_id"]].merge(
            s[["audio_id", "reference_id", "fold_S2"]],
            on="audio_id",
            how="outer",
            suffixes=("_manifest", "_split"),
            validate="one_to_one",
            indicator=True,
        )

        if not (merged["_merge"] == "both").all():
            raise ValueError(
                "Manifest i splits no contenen els mateixos audio_id."
            )

        if not (
            merged["reference_id_manifest"]
            == merged["reference_id_split"]
        ).all():
            raise ValueError(
                "Els reference_id no coincideixen entre manifest i splits."
            )

        if merged["fold_S2"].isna().any():
            raise ValueError("Hi ha mostres sense fold S2.")

        mapping = merged.groupby("reference_id_manifest")[
            "fold_S2"
        ].nunique()

        if (mapping != 1).any():
            raise ValueError(
                "Una mateixa referència apareix en diversos folds."
            )

        reverse = merged.groupby("fold_S2")[
            "reference_id_manifest"
        ].nunique()

        if (reverse != 1).any():
            print(
                "AVÍS: algun fold conté més d'una referència. "
                "No és necessàriament incorrecte, però difereix "
                "del protocol original d'un fold per referència."
            )

        print("Assignació S2 coherent amb el manifest.")
        print(
            merged.groupby("fold_S2")
            .size()
            .to_string()
        )
    else:
        print("No s'ha trobat splits_v1.csv.")

    print("\nAuditoria finalitzada.")
    print(f"Resultats disponibles a: {OUT_DIR}")


if __name__ == "__main__":
    main()
