"""
Extreu embeddings acústics dels àudios EMIS mitjançant el model preentrenat WavLM Base Plus, 
remostrejant els àudios de 24 kHz a 16 kHz.

Calcula la mitjana i la desviació estàndard de les representacions de totes les capes del model 
per capturar informació acústica i contextual de la veu.

Comprova la qualitat dels embeddings detectant valors absents, infinits i dimensions constants.

Genera `outputs/features/wavlm_base_plus_meanstd_sr16000.npz`

Desa els embeddings de cada àudio (`audio_id`), amb la mitjana (`mean`) i la desviació estàndard 
(`std`) de cada capa.

Serveix per entrenar i avaluar classificadors de predicció emocional basats en representacions WavLM.

Genera `outputs/features/wavlm_base_plus_meanstd_sr16000.json`

Desa les metadades de l'extracció, el model utilitzat, les dimensions dels embeddings, 
les freqüències de mostreig i els hashes SHA-256.

Serveix per garantir la traçabilitat i reproductibilitat de l'extracció.
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

# Arrel del projecte: dos nivells per sobre de la carpeta on hi ha aquest script
ROOT = Path(__file__).resolve().parents[2]
# Freqüència de mostreig que espera WavLM (16 kHz)
TARGET_SR = 16000


def read_mono(zf: zipfile.ZipFile, member: str):
    """Llegeix un àudio directament des del ZIP (sense descomprimir-lo al disc)
    i el converteix a mono fent la mitjana dels canals. Retorna (senyal, sr)."""
    # always_2d=True garanteix forma (mostres, canals) tant si és mono com estèreo
    x, sr = sf.read(io.BytesIO(zf.read(member)), always_2d=True, dtype="float32")
    return x.mean(axis=1).astype("float32"), sr


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """Remostreja el senyal a 16 kHz (p. ex. de 24 kHz) amb un filtre polifàsic."""
    # Si ja està a 16 kHz, no cal fer res
    if sr == TARGET_SR:
        return x
    # Factors de remostratge enters mínims: up = 16000/g, down = sr/g
    g = math.gcd(TARGET_SR, sr)
    return resample_poly(x, TARGET_SR // g, sr // g).astype("float32")


@torch.inference_mode()  # desactiva el càlcul de gradients: més ràpid i menys memòria
def embed(fe, model, x16: np.ndarray, device: str):
    """Calcula la mitjana i la desviació estàndard temporals de cada capa de WavLM
    per a un únic àudio. Retorna (mean, std, nombre_de_frames)."""
    # El feature extractor prepara l'entrada (p. ex. normalització) i la converteix a tensor
    inp = fe(x16, sampling_rate=TARGET_SR, return_tensors="pt")
    # output_hidden_states=True retorna la sortida de totes les capes, no només l'última
    hs = model(inp.input_values.to(device), output_hidden_states=True).hidden_states
    # Apila les capes i elimina la dimensió de batch (només hi ha 1 àudio)
    H = torch.stack(hs, dim=0)[:, 0].float()  # (L, T, D): capes, frames, dimensions
    # Agrega al llarg del temps (dim=1). unbiased=False -> desviació poblacional (ddof=0)
    return (H.mean(dim=1).cpu().numpy(), H.std(dim=1, unbiased=False).cpu().numpy(),
            H.shape[1])


def main() -> None:
    # ---------------- Arguments de línia d'ordres ----------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ROOT / "data/raw/audios_EMIS.zip")              # àudios originals
    ap.add_argument("--manifest", default=ROOT / "data/processed/emis_manifest.csv")  # llista d'àudios
    ap.add_argument("--out-dir", default=ROOT / "outputs/features")                   # carpeta de sortida
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")                   # model de HuggingFace
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")  # GPU si n'hi ha
    ap.add_argument("--expect-n", type=int, default=1248)  # nombre d'àudios esperat (control)
    a = ap.parse_args()

    # ---------------- Càrrega del manifest i del model ----------------
    # dtype=str evita que pandas interpreti els identificadors com a números
    man = pd.read_csv(a.manifest, dtype=str)
    # Comprovacions de seguretat: nombre esperat de clips i identificadors únics
    assert len(man) == a.expect_n and man["audio_id"].is_unique
    fe = AutoFeatureExtractor.from_pretrained(a.model)
    # Model en mode avaluació (desactiva dropout, etc.) i mogut al dispositiu escollit
    model = WavLMModel.from_pretrained(a.model).to(a.device).eval()
    # Comprova que el model espera realment 16 kHz
    assert fe.sampling_rate == TARGET_SR
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- Extracció d'embeddings ----------------
    # srs: conjunt de freqüències d'entrada vistes; secs: durades; frames: frames per clip
    means, stds, srs, secs, frames, t0 = [], [], set(), [], [], time.time()
    with zipfile.ZipFile(a.zip) as zf:
        for i, m in enumerate(man["zip_member"]):
            x, sr = read_mono(zf, m)       # llegeix l'àudio i el passa a mono
            srs.add(sr)                    # registra la freqüència original
            x16 = to_16k(x, sr)            # remostreja a 16 kHz
            secs.append(len(x16) / TARGET_SR)  # durada en segons
            mu, sd, T = embed(fe, model, x16, a.device)
            means.append(mu)
            stds.append(sd)
            frames.append(T)
            # Informe de progrés cada 100 àudios
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(man)}  ({time.time() - t0:.0f} s)")
    elapsed = time.time() - t0
    # Apila tots els clips: M i S tenen forma (N clips, L capes, D dimensions)
    M, S = np.stack(means).astype("float32"), np.stack(stds).astype("float32")  # (N, L, D)

    # ---------------- Desa embeddings (.npz) ----------------
    # Nom base a partir del model, p. ex. "wavlm_base_plus"
    tag = a.model.split("/")[-1].replace("-", "_")
    npz = out / f"{tag}_meanstd_sr{TARGET_SR}.npz"
    np.savez_compressed(npz, audio_id=man["audio_id"].to_numpy().astype(str), mean=M, std=S)

    # ---------------- Desa metadades (.json) per a la traçabilitat ----------------
    meta = {"model": a.model, "model_commit": getattr(model.config, "_commit_hash", None),
            "transformers": transformers.__version__, "torch": torch.__version__,
            "device": a.device, "input_sample_rates": sorted(srs), "model_sample_rate": TARGET_SR,
            "feature_extractor_do_normalize": bool(getattr(fe, "do_normalize", False)),
            "pooling": ["mean", "std (ddof=0)"], "n_clips": int(M.shape[0]),
            "n_layers_incl_cnn_proj": int(M.shape[1]), "hidden_size": int(M.shape[2]),
            "seconds_total": round(elapsed, 1),
            "clip_seconds": {"min": round(min(secs), 2), "median": round(float(np.median(secs)), 2),
                             "max": round(max(secs), 2)},
            # Hashes SHA-256 per verificar que manifest i features no han canviat
            "manifest_sha256": hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
            "features_sha256": hashlib.sha256(npz.read_bytes()).hexdigest()}
    (out / f"{tag}_meanstd_sr{TARGET_SR}.json").write_text(json.dumps(meta, indent=2))

    # ---------------- QC (control de qualitat) ----------------
    # Resum general de l'extracció
    print(f"\nExtracció: {M.shape[0]} clips | {M.shape[1]} capes x {M.shape[2]} dims | {elapsed:.0f} s"
          f" | sr entrada: {sorted(srs)} -> {TARGET_SR}")
    print(f"Durada (s): min {min(secs):.2f} | mediana {np.median(secs):.2f} | max {max(secs):.2f}")
    # Valors absents (NaN) i infinits, tant a la mitjana com a la desviació
    print(f"NaN: {int(np.isnan(M).sum() + np.isnan(S).sum())} | "
          f"inf: {int(np.isinf(M).sum() + np.isinf(S).sum())}")
    # Clips amb l'embedding mean completament a zero (senyal d'error)
    print(f"Files amb tots els valors a 0 (mean): {int((np.abs(M).sum(axis=(1, 2)) == 0).sum())}")
    # Dimensions sense variança entre clips (desviació 0): no aporten informació
    const = int((M.reshape(len(M), -1).std(axis=0) == 0).sum())
    print(f"Dimensions constants (mean, totes les capes): {const}")
    # Norma L2 de l'embedding mean, promitjada entre clips, per a cada capa
    print("Norma L2 mitjana de l'embedding mean per capa:")
    print(np.round(np.linalg.norm(M, axis=2).mean(axis=0), 2))
    print(f"\nFeatures: {npz}")


if __name__ == "__main__":
    main()