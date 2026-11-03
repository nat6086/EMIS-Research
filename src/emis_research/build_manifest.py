"""
Construeix el manifest d'EMIS: una fila per cada enregistrament d'àudio.

Què fa aquest script, en paraules senzilles:
  1. Obre el ZIP amb els fitxers d'àudio i llegeix el NOM de cada fitxer
     (el nom codifica qui/què/com s'ha generat l'àudio).
  2. Busca la frase que es diu a l'àudio dins del CSV de textos.
  3. Ho desa tot en una sola taula: data/processed/emis_manifest.csv
  
data/processed/emis_manifest.csv s'utilitza en la resta dels experiments.
"""

import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# On són els fitxers (les rutes parteixen de la carpeta arrel del projecte)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
ZIP_PATH = ROOT / "data" / "raw" / "audios_EMIS.zip"           # els àudios
CSV_PATH = ROOT / "data" / "raw" / "text_samples_EMIS.csv"     # les frases
OUTPUT_PATH = ROOT / "data" / "processed" / "emis_manifest.csv"  # el que creem

# Les columnes de la taula final, en l'ordre en què es desaran
COLUMNS = [
    "audio_id", "zip_member", "text", "generation_emotion", "reference_id",
    "tts_system", "text_id", "text_category", "text_emotion", "text_cue_type",
]


def parse_filename(path: str) -> dict:
    """
    Llegeix la informació amagada dins del nom d'un fitxer d'àudio.

    Hi ha dos tipus de noms:
      6 parts: <text_id>_<emotion>_<cue>_<voice>_<reference_id>_<tts>
               p. ex. 0012_angry_explicit_happy_ref03_COSY
      5 parts: <text_id>_neutral_<voice>_<reference_id>_<tts>
               p. ex. 0012_neutral_sad_ref01_F5TTS
    """
    # Traiem el ".wav" i la carpeta, i tallem el nom a cada "_"
    parts = Path(path).stem.split("_")

    # Si no té 5 o 6 trossos, no és un nom vàlid d'EMIS
    if len(parts) not in (5, 6):
        raise ValueError(f"Unexpected filename: {path}")

    # El primer tros és sempre el número de text, i els tres últims són sempre
    # veu / referència / TTS. El que queda al mig és la categoria del text
    # (1 tros per a neutral, 2 trossos per a emoció + tipus d'indicació).
    text_id, *category, voice, reference_id, tts = parts

    return {
        "text_id": text_id,
        "text_category": "_".join(category),   # p. ex. "angry_explicit" o "neutral"
        "text_emotion": category[0],           # p. ex. "angry" o "neutral"
        # Els textos neutrals no tenen tipus d'indicació; l'etiquetem "neutral"
        "text_cue_type": category[1] if len(category) == 2 else "neutral",
        "generation_emotion": voice,           # emoció amb què s'ha generat l'àudio
        "reference_id": reference_id,          # referència acústica utilitzada
        "tts_system": tts,                     # COSY, F5TTS o STYLE
    }


def main() -> None:
    # Carreguem la taula de frases (una fila per text, una columna per categoria)
    texts = pd.read_csv(CSV_PATH)

    # Llistem tots els .wav del ZIP (ordenats perquè la sortida sigui estable)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        members = sorted(n for n in archive.namelist() if n.lower().endswith(".wav"))

    records = []
    for member in members:
        # Pas 1: obtenim les metadades a partir del nom del fitxer
        meta = parse_filename(member)

        # Pas 2: busquem la frase que es diu.
        # El text_id comença a 1 però les files de la taula comencen a 0,
        # d'aquí el "- 1". La columna és la categoria (p. ex. "angry_explicit").
        text = texts.at[int(meta["text_id"]) - 1, meta["text_category"]]

        # Ens aturem si la frase està buida: seria un forat a les dades
        if pd.isna(text) or not str(text).strip():
            raise ValueError(f"Missing text: {member}")

        # Pas 3: desem una fila per a aquest enregistrament
        records.append({
            "audio_id": Path(member).stem,   # nom del fitxer sense ".wav"
            "zip_member": member,            # ruta completa dins del ZIP
            "text": str(text),
            **meta,                          # afegim totes les metadades del nom
        })

    # Convertim la llista de files en una taula amb les columnes en l'ordre correcte
    manifest = pd.DataFrame(records)[COLUMNS]

    # -----------------------------------------------------------------------
    # Comprovacions de seguretat: si alguna falla, l'script s'atura i diu per què
    # -----------------------------------------------------------------------
    # 1) Esperem exactament 1248 àudios
    assert len(manifest) == 1248, f"Expected 1248 recordings, found {len(manifest)}"
    # 2) Cap àudio pot aparèixer dues vegades
    assert not manifest["audio_id"].duplicated().any(), "Duplicate audio identifiers"
    # 3) Cada text ha de tenir 12 àudios (4 emocions x 3 sistemes TTS)
    sizes = manifest.groupby(["text_category", "text_id"]).size()
    assert (sizes == 12).all(), "Some text groups don't have 12 recordings"

    # Creem la carpeta de sortida si cal i desem el CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(manifest)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()