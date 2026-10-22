"""Extracció d'embeddings WavLM (totes les capes, pooling mitjana + desviació) + QC.

Per executar:
& c:/Users/nmord/EMIS-Research/.venv/Scripts/python.exe c:/Users/nmord/EMIS-Research/src/emis_research/wavlm_embeddings.py

Només lectura sobre l'àudio: cap normalització pròpia, retall ni filtratge. Es llegeix el WAV
del ZIP a 24 kHz, es passa a mono si cal i es remostra a 16 kHz (WavLM ho exigeix; és l'únic
canvi). La normalització d'entrada la decideix el feature extractor del propi model.
Es processa un clip per vegada (sense padding): WavLM base no usa attention mask i el padding
alteraria els embeddings.
Sortida: NPZ amb audio_id + mean (N, L, D) + std (N, L, D) i un JSON amb la traçabilitat.
L = 1 + nombre de capes transformer (índex 0 = sortida de la projecció CNN).
"""
import argparse
import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import transformers
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, WavLMModel

ROOT = Path(__file__).resolve().parents[2]
TARGET_SR = 16000


def read_mono(zf: zipfile.ZipFile, member: str):
    x, sr = sf.read(io.BytesIO(zf.read(member)), always_2d=True, dtype="float32")
    return x.mean(axis=1).astype("float32"), sr


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return x
    g = math.gcd(TARGET_SR, sr)
    return resample_poly(x, TARGET_SR // g, sr // g).astype("float32")


@torch.inference_mode()
def embed(fe, model, x16: np.ndarray, device: str):
    inp = fe(x16, sampling_rate=TARGET_SR, return_tensors="pt")
    hs = model(inp.input_values.to(device), output_hidden_states=True).hidden_states
    H = torch.stack(hs, dim=0)[:, 0].float()  # (L, T, D)
    return (H.mean(dim=1).cpu().numpy(), H.std(dim=1, unbiased=False).cpu().numpy(),
            H.shape[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ROOT / "data/raw/audios_EMIS.zip")
    ap.add_argument("--manifest", default=ROOT / "data/processed/emis_manifest.csv")
    ap.add_argument("--out-dir", default=ROOT / "outputs/features")
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--expect-n", type=int, default=1248)
    a = ap.parse_args()

    man = pd.read_csv(a.manifest, dtype=str)
    assert len(man) == a.expect_n and man["audio_id"].is_unique
    fe = AutoFeatureExtractor.from_pretrained(a.model)
    model = WavLMModel.from_pretrained(a.model).to(a.device).eval()
    assert fe.sampling_rate == TARGET_SR
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    means, stds, srs, secs, frames, t0 = [], [], set(), [], [], time.time()
    with zipfile.ZipFile(a.zip) as zf:
        for i, m in enumerate(man["zip_member"]):
            x, sr = read_mono(zf, m)
            srs.add(sr)
            x16 = to_16k(x, sr)
            secs.append(len(x16) / TARGET_SR)
            mu, sd, T = embed(fe, model, x16, a.device)
            means.append(mu)
            stds.append(sd)
            frames.append(T)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(man)}  ({time.time() - t0:.0f} s)")
    elapsed = time.time() - t0
    M, S = np.stack(means).astype("float32"), np.stack(stds).astype("float32")  # (N, L, D)

    tag = a.model.split("/")[-1].replace("-", "_")
    npz = out / f"{tag}_meanstd_sr{TARGET_SR}.npz"
    np.savez_compressed(npz, audio_id=man["audio_id"].to_numpy().astype(str), mean=M, std=S)
    meta = {"model": a.model, "model_commit": getattr(model.config, "_commit_hash", None),
            "transformers": transformers.__version__, "torch": torch.__version__,
            "device": a.device, "input_sample_rates": sorted(srs), "model_sample_rate": TARGET_SR,
            "feature_extractor_do_normalize": bool(getattr(fe, "do_normalize", False)),
            "pooling": ["mean", "std (ddof=0)"], "n_clips": int(M.shape[0]),
            "n_layers_incl_cnn_proj": int(M.shape[1]), "hidden_size": int(M.shape[2]),
            "seconds_total": round(elapsed, 1),
            "clip_seconds": {"min": round(min(secs), 2), "median": round(float(np.median(secs)), 2),
                             "max": round(max(secs), 2)},
            "manifest_sha256": hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
            "features_sha256": hashlib.sha256(npz.read_bytes()).hexdigest()}
    (out / f"{tag}_meanstd_sr{TARGET_SR}.json").write_text(json.dumps(meta, indent=2))

    # ---------------- QC ----------------
    print(f"\nExtracció: {M.shape[0]} clips | {M.shape[1]} capes x {M.shape[2]} dims | {elapsed:.0f} s"
          f" | sr entrada: {sorted(srs)} -> {TARGET_SR}")
    print(f"Durada (s): min {min(secs):.2f} | mediana {np.median(secs):.2f} | max {max(secs):.2f}")
    print(f"NaN: {int(np.isnan(M).sum() + np.isnan(S).sum())} | "
          f"inf: {int(np.isinf(M).sum() + np.isinf(S).sum())}")
    print(f"Files amb tots els valors a 0 (mean): {int((np.abs(M).sum(axis=(1, 2)) == 0).sum())}")
    const = int((M.reshape(len(M), -1).std(axis=0) == 0).sum())
    print(f"Dimensions constants (mean, totes les capes): {const}")
    print("Norma L2 mitjana de l'embedding mean per capa:")
    print(np.round(np.linalg.norm(M, axis=2).mean(axis=0), 2))
    print(f"\nFeatures: {npz}")


if __name__ == "__main__":
    main()
