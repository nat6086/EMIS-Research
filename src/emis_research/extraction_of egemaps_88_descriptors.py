"""Extracció d'eGeMAPSv02 (Functionals, 88 descriptors) amb openSMILE + QC de features.

Per executar:
& c:/Users/nmord/EMIS-Research/.venv/Scripts/python.exe c:/Users/nmord/EMIS-Research/src/emis_research/egemaps_88_descriptors.py --sr16-check

Només lectura sobre l'àudio: cap normalització, retall ni filtratge. Es llegeix el WAV
del ZIP a la seva freqüència nativa (24 kHz) i es converteix a mono només si cal.
Sortida: CSV amb audio_id + 88 columnes, i un JSON amb la traçabilitat.
Opcional (--sr16-check): robustesa 24 kHz vs 16 kHz sobre un subconjunt estratificat.
"""
import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import opensmile
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
KEY_FEATS = ["F0semitoneFrom27.5Hz_sma3nz_amean", "loudness_sma3_amean", "VoicedSegmentsPerSec"]


def make_smile() -> opensmile.Smile:
    return opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                           feature_level=opensmile.FeatureLevel.Functionals)


def read_mono(zf: zipfile.ZipFile, member: str):
    x, sr = sf.read(io.BytesIO(zf.read(member)), always_2d=True, dtype="float32")
    return x.mean(axis=1).astype("float32"), sr


def extract(smile, x: np.ndarray, sr: int) -> np.ndarray:
    return smile.process_signal(x, sr).iloc[0].to_numpy(dtype="float64")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ROOT / "data/raw/audios_EMIS.zip")
    ap.add_argument("--manifest", default=ROOT / "data/processed/emis_manifest.csv")
    ap.add_argument("--out-dir", default=ROOT / "outputs/features")
    ap.add_argument("--sr16-check", action="store_true",
                    help="compara 24 kHz natiu vs 16 kHz en un subconjunt estratificat")
    ap.add_argument("--per-cell", type=int, default=10, help="clips per TTS x emoció (sr16-check)")
    ap.add_argument("--expect-n", type=int, default=1248)
    a = ap.parse_args()

    man = pd.read_csv(a.manifest, dtype=str)
    assert len(man) == a.expect_n and man["audio_id"].is_unique
    smile = make_smile()
    names = list(smile.feature_names)
    assert len(names) == 88, f"eGeMAPSv02 hauria de tenir 88 descriptors, té {len(names)}"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, srs, t0 = [], set(), time.time()
    with zipfile.ZipFile(a.zip) as zf:
        for i, m in enumerate(man["zip_member"]):
            x, sr = read_mono(zf, m)
            srs.add(sr)
            rows.append(extract(smile, x, sr))
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(man)}  ({time.time() - t0:.0f} s)")
    elapsed = time.time() - t0
    F = pd.DataFrame(rows, columns=names)
    F.insert(0, "audio_id", man["audio_id"].to_numpy())
    sr_tag = "sr" + "-".join(str(s) for s in sorted(srs))
    csv = out / f"egemaps_v02_functionals_{sr_tag}.csv"
    F.to_csv(csv, index=False)
    meta = {"opensmile": opensmile.__version__, "feature_set": "eGeMAPSv02",
            "feature_level": "Functionals", "sample_rates": sorted(srs), "n_clips": len(F),
            "n_features": len(names), "seconds": round(elapsed, 1),
            "manifest_sha256": hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
            "features_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
            "feature_names": names}
    (out / f"egemaps_v02_functionals_{sr_tag}.json").write_text(json.dumps(meta, indent=2))

    # ---------------- QC de features ----------------
    X = F[names]
    print(f"\nExtracció: {len(F)} clips x {len(names)} descriptors en {elapsed:.0f} s | sr: {sorted(srs)}"
          f" | openSMILE {opensmile.__version__}")
    print(f"NaN: {int(X.isna().sum().sum())} | inf: {int(np.isinf(X.to_numpy()).sum())}")
    const = [c for c in names if X[c].nunique(dropna=True) <= 1]
    print(f"Columnes constants: {len(const)} {const[:10]}")
    z = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    ext = (z.abs() > 6).sum()
    print("Columnes amb valors |z|>6 (nombre de clips; heurístic, només descriptiu):")
    print(ext[ext > 0].sort_values(ascending=False).head(10).to_string() or "  cap")
    print(f"Files amb tots els descriptors a 0: {int((X.abs().sum(axis=1) == 0).sum())}")
    if "VoicedSegmentsPerSec" in X:
        print(f"Clips sense cap segment sonor (VoicedSegmentsPerSec==0): {int((X['VoicedSegmentsPerSec'] == 0).sum())}")

    d = man[["audio_id", "tts_system", "generation_emotion"]].merge(F, on="audio_id")
    for f in [k for k in KEY_FEATS if k in names]:
        print(f"\n{f} — mitjana per TTS x emoció:")
        print(d.groupby(["tts_system", "generation_emotion"])[f].mean().round(3).unstack().to_string())
    if KEY_FEATS[0] in names:
        hz = 27.5 * 2 ** (d[KEY_FEATS[0]] / 12)
        print("\nF0 mitjana aproximada en Hz (només descriptiu; 0 st = sense F0 detectat):")
        print(hz.groupby([d["tts_system"], d["generation_emotion"]]).mean().round(1).unstack().to_string())

    # ---------------- robustesa 24 kHz vs 16 kHz ----------------
    if a.sr16_check:
        sub = (man.groupby(["tts_system", "generation_emotion"], group_keys=False)
                  .apply(lambda g: g.sample(min(a.per_cell, len(g)), random_state=SEED)))
        r24, r16 = [], []
        with zipfile.ZipFile(a.zip) as zf:
            for m in sub["zip_member"]:
                x, sr = read_mono(zf, m)
                r24.append(extract(smile, x, sr))
                x16 = resample_poly(x, 16000, sr).astype("float32") if sr != 16000 else x
                r16.append(extract(smile, x16, 16000))
        A, B = np.array(r24), np.array(r16)
        rho = pd.Series([spearmanr(A[:, j], B[:, j]).correlation for j in range(A.shape[1])], index=names)
        rho.to_csv(out / "sr16_vs_native_spearman.csv", header=["spearman"])
        print(f"\nRobustesa sr (n={len(sub)} clips): Spearman 24k vs 16k per descriptor")
        print(rho.describe().round(3).to_string())
        print("10 descriptors menys estables:")
        print(rho.sort_values().head(10).round(3).to_string())
    print(f"\nFeatures: {csv}")


if __name__ == "__main__":
    main()