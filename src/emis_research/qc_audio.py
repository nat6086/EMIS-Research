"""QC d'àudio d'EMIS (només lectura). No modifica ni elimina cap mostra.

Calcula per fitxer: sr, canals, durada, pic, fracció de clipping, RMS (dBFS),
offset DC, silenci inicial/final i fracció de frames de baixa energia.
Les banderes són heurístiques descriptives, NO criteris d'exclusió.
"""
import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-10
FRAME, HOP = 0.025, 0.010          # finestra 25 ms, salt 10 ms
SIL_DBFS = -50.0                   # llindar de "silenci" per frame (heurístic)
CLIP_LEVEL = 0.999                 # |x| >= això compta com a possible clipping


def frame_rms_db(x: np.ndarray, sr: int) -> np.ndarray:
    n, h = int(FRAME * sr), int(HOP * sr)
    if len(x) < n:
        return np.array([20 * np.log10(np.sqrt(np.mean(x ** 2)) + EPS)])
    idx = np.arange(0, len(x) - n + 1, h)
    return np.array([20 * np.log10(np.sqrt(np.mean(x[i:i + n] ** 2)) + EPS) for i in idx])


def analyse(x: np.ndarray, sr: int, ch: int) -> dict:
    mono = x.mean(axis=1) if x.ndim > 1 else x
    db = frame_rms_db(mono, sr)
    voiced = np.where(db > SIL_DBFS)[0]
    lead = voiced[0] * HOP if len(voiced) else len(mono) / sr
    trail = (len(db) - 1 - voiced[-1]) * HOP if len(voiced) else len(mono) / sr
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    return {
        "sample_rate": sr, "channels": ch, "duration_s": len(mono) / sr,
        "peak": peak, "clip_frac": float(np.mean(np.abs(mono) >= CLIP_LEVEL)),
        "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(mono ** 2)) + EPS)),
        "dc_offset": float(np.mean(mono)), "lead_sil_s": float(lead),
        "trail_sil_s": float(trail), "low_energy_frac": float(np.mean(db <= SIL_DBFS)),
        "has_nan_inf": bool(~np.isfinite(mono).all()),
    }


def eta_squared(df: pd.DataFrame, value: str, group: str) -> float:
    g = df.groupby(group)[value]
    ss_b = (g.count() * (g.mean() - df[value].mean()) ** 2).sum()
    ss_t = ((df[value] - df[value].mean()) ** 2).sum()
    return float(ss_b / ss_t) if ss_t > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ROOT / "data/raw/audios_EMIS.zip")
    ap.add_argument("--manifest", default=ROOT / "data/processed/emis_manifest.csv")
    ap.add_argument("--out", default=ROOT / "outputs/qc")
    a = ap.parse_args()
    man = pd.read_csv(a.manifest, dtype=str)
    rows = []
    with zipfile.ZipFile(a.zip) as z:
        for m in man["zip_member"]:
            try:
                x, sr = sf.read(io.BytesIO(z.read(m)), always_2d=True, dtype="float64")
                rec = analyse(x, sr, x.shape[1]) | {"zip_member": m, "read_error": ""}
            except Exception as e:  # noqa: BLE001
                rec = {"zip_member": m, "read_error": str(e)[:80]}
            rows.append(rec)
    q = man.merge(pd.DataFrame(rows), on="zip_member", validate="one_to_one")
    ok = q["read_error"].eq("")
    q["flag_clip"] = q["clip_frac"] > 0
    q["flag_peak_high"] = q["peak"] >= CLIP_LEVEL
    q["flag_long_edge_sil"] = (q["lead_sil_s"] > 0.5) | (q["trail_sil_s"] > 0.5)
    q["flag_low_level"] = q["rms_dbfs"] < -40
    q["flag_nonfinite"] = q["has_nan_inf"].fillna(False)
    q["flag_sr_not_24k"] = ok & (q["sample_rate"] != 24000)
    q["flag_not_mono"] = ok & (q["channels"] != 1)
    flags = [c for c in q.columns if c.startswith("flag_")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    q.to_csv(out / "audio_qc.csv", index=False)

    print(f"Fitxers: {len(q)} | errors de lectura: {(~ok).sum()}")
    print("\nBanderes (nombre de fitxers):")
    print(q[flags].sum().to_string())
    cols = ["duration_s", "peak", "rms_dbfs", "lead_sil_s", "trail_sil_s", "low_energy_frac"]
    print("\nResum global:\n", q.loc[ok, cols].describe().round(3).to_string())
    print("\nPer TTS x emoció de generació (mitjanes):")
    print(q[ok].groupby(["tts_system", "generation_emotion"])[["duration_s", "rms_dbfs"]]
          .mean().round(3).unstack().to_string())
    print("\neta² de l'emoció de generació, dins de cada TTS (descriptiu; mostres agrupades per text):")
    for tts, d in q[ok].groupby("tts_system"):
        print(f"  {tts}: durada={eta_squared(d, 'duration_s', 'generation_emotion'):.3f} "
              f"rms_dbfs={eta_squared(d, 'rms_dbfs', 'generation_emotion'):.3f}")
    print(f"\nCSV: {out / 'audio_qc.csv'}")


if __name__ == "__main__":
    main()