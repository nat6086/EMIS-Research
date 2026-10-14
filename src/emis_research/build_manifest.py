
from pathlib import Path
from collections import Counter
import zipfile

import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = ROOT / "data" / "raw" / "audios_EMIS.zip"
CSV_PATH = ROOT / "data" / "raw" / "text_samples_EMIS.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "emis_manifest.csv"

EXPECTED_EMOTIONS = {"angry", "happy", "sad", "neutral"}
EXPECTED_TTS = {"COSY", "F5TTS", "STYLE"}


def parse_filename(filename: str) -> dict:
    """Extract metadata from an EMIS audio filename."""
    parts = Path(filename).stem.split("_")

    if len(parts) == 6:
        text_id, emotion, expression, voice, reference_id, tts = parts
        text_category = f"{emotion}_{expression}"

    elif len(parts) == 5:
        text_id, text_category, voice, reference_id, tts = parts

        if text_category != "neutral":
            raise ValueError(f"Unexpected filename: {filename}")
    else:
        raise ValueError(f"Unexpected filename: {filename}")

    if voice not in EXPECTED_EMOTIONS:
        raise ValueError(f"Unknown generation emotion: {filename}")

    if tts not in EXPECTED_TTS:
        raise ValueError(f"Unknown TTS system: {filename}")

    if not text_id.isdigit():
        raise ValueError(f"Invalid text ID: {filename}")

    return {
        "text_id": text_id,
        "text_category": text_category,
        "generation_emotion": voice,
        "reference_id": reference_id,
        "tts_system": tts,
    }


def build_manifest() -> pd.DataFrame:
    """Build the audio-text manifest from original EMIS files."""
    texts = pd.read_csv(CSV_PATH)
    records = []

    with zipfile.ZipFile(ZIP_PATH) as archive:
        filenames = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".wav")
        )

        for filename in filenames:
            metadata = parse_filename(Path(filename).name)

            category = metadata["text_category"]
            text_index = int(metadata["text_id"]) - 1

            if category not in texts.columns:
                raise ValueError(f"Unknown text category: {category}")

            if not 0 <= text_index < len(texts):
                raise ValueError(f"Invalid text index: {filename}")

            text = texts.iloc[text_index][category]

            if pd.isna(text) or not str(text).strip():
                raise ValueError(f"Missing text: {filename}")

            records.append({
                "audio_id": Path(filename).stem,
                "zip_member": filename,
                "text": str(text),
                **metadata,
            })

    manifest = pd.DataFrame(records)

    if len(manifest) != 1248:
        raise ValueError(
            f"Expected 1248 recordings, found {len(manifest)}"
        )

    if manifest["audio_id"].duplicated().any():
        raise ValueError("Duplicate audio identifiers detected")

    # Check factorial completeness for each text.
    group_columns = ["text_category", "text_id"]
    expected_pairs = {
        (emotion, tts)
        for emotion in EXPECTED_EMOTIONS
        for tts in EXPECTED_TTS
    }

    for group_key, group in manifest.groupby(group_columns):
        pairs = list(
            zip(group["generation_emotion"], group["tts_system"])
        )

        if len(pairs) != 12 or set(pairs) != expected_pairs:
            raise ValueError(f"Incomplete factorial group: {group_key}")

        if len(set(pairs)) != 12:
            raise ValueError(f"Duplicate factorial pairs: {group_key}")

    return manifest


def main() -> None:
    manifest = build_manifest()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("Manifest:", OUTPUT_PATH)
    print("Total recordings:", len(manifest))

    unique_texts = manifest[
        ["text_category", "text_id"]
    ].drop_duplicates()

    print("Unique text groups:", len(unique_texts))

    print("\nTTS distribution:")
    print(manifest["tts_system"].value_counts().to_string())

    print("\nGeneration emotion distribution:")
    print(manifest["generation_emotion"].value_counts().to_string())

    print("\nMissing values:")
    print(manifest.isna().sum().to_string())

    print("\nManifest validation: OK")


if __name__ == "__main__":
    main()
