"""
Extreu els 88 descriptors acústics eGeMAPSv02 de cada àudio EMIS mitjançant openSMILE, 
mantenint la freqüència de mostreig original (24 kHz).

Genera outputs/features/egemaps_v02_functionals_sr24000.csv

Desa els 88 descriptors acústics de cada àudio juntament amb el seu identificador (audio_id).

S'utilitza per entrenar i avaluar els classificadors de predicció emocional.
"""
import argparse
import io
import zipfile
from pathlib import Path

import opensmile
import pandas as pd
import soundfile as sf

# Arrel del projecte: dos nivells per sobre d'aquest fitxer (src/emis_research/ -> arrel)
ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # --- Arguments de línia d'ordres (rutes per defecte dins del projecte) ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ROOT / "data/raw/audios_EMIS.zip")                  # ZIP amb els àudios
    ap.add_argument("--manifest", default=ROOT / "data/processed/emis_manifest.csv")     # llista d'àudios (audio_id, zip_member)
    ap.add_argument("--out-dir", default=ROOT / "outputs/features")                      # carpeta de sortida
    a = ap.parse_args()

    # Llegim el manifest com a text (dtype=str) per no alterar els identificadors
    man = pd.read_csv(a.manifest, dtype=str)

    # Configurem openSMILE: conjunt eGeMAPSv02 amb estadístics globals per àudio (Functionals)
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)

    rows, srs = [], set()  # rows: vector de descriptors per àudio; srs: freqüències de mostreig trobades
    with zipfile.ZipFile(a.zip) as zf:
        for i, member in enumerate(man["zip_member"], 1):
            # Llegim l'àudio directament de dins del ZIP, sense descomprimir-lo al disc
            x, sr = sf.read(io.BytesIO(zf.read(member)), always_2d=True, dtype="float32")
            srs.add(sr)  # guardem la freqüència de mostreig (es manté la nativa, sense remostrejar)
            # Passem a mono (mitjana dels canals), extraiem els 88 descriptors i els afegim a la llista
            rows.append(smile.process_signal(x.mean(axis=1), sr).iloc[0].to_numpy(dtype="float64"))
            if i % 200 == 0:  # indicador de progrés cada 200 àudios
                print(f"  {i}/{len(man)}")

    # Construïm la taula: una fila per àudio, una columna per descriptor
    F = pd.DataFrame(rows, columns=smile.feature_names)
    F.insert(0, "audio_id", man["audio_id"].to_numpy())  # identificador a la primera columna

    # Creem la carpeta de sortida si no existeix
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # El nom del fitxer inclou la freqüència de mostreig (p. ex. sr24000)
    csv = out / f"egemaps_v02_functionals_sr{'-'.join(map(str, sorted(srs)))}.csv"
    F.to_csv(csv, index=False)
    print(f"{len(F)} clips x {F.shape[1] - 1} descriptors -> {csv}")


if __name__ == "__main__":
    main()