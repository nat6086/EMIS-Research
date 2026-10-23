"""Extracció de sortides de models d'emoció textual sobre els textos ÚNICS d'EMIS (sense fine-tuning).

Per executar (exemple):
python src/emis_research/text_emotion_extraction.py --models hartmann_large hartmann_distil go_emotions deberta_zeroshot qwen3_4b

Restriccions (anti data-leakage):
  * L'únic input de cada model és la cadena literal `text`. Cap altra columna entra a la inferència.
  * Cada frase única (cadena exacta) s'avalua una sola vegada, encara que surti en molts àudios.
  * El mapatge model -> {angry, happy, neutral, sad} i els prompts estan fixats AQUÍ, a priori.
    Canviar-los després de veure resultats sobre EMIS és ajust post-hoc i s'ha de reportar com a tal.
  * Les revisions (commit SHA) dels checkpoints es resolen a la primera execució i es desen a
    model_revisions.json; les execucions posteriors les reutilitzen (commiteja aquest fitxer).

Sortida (outputs/text_baseline/):
  features/<model>.csv   text, text_category, text_id (només traçabilitat), p_angry..p_sad (sumen 1),
                         raw_* (sortides natives del model)
  run_meta.json          versions, checkpoints, revisions, prompts, mapatges, seed, dispositiu
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub import HfApi
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer)

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
EMOS = ["angry", "happy", "neutral", "sad"]
EPS = 1e-9

# ---------------- Mapatges i prompts (fixats a priori) ----------------
NLI_TEMPLATE = "The speaker feels {}."          # hipòtesi genèrica; paraules = les 4 etiquetes
LLM_SYSTEM = "You are an expert in emotion analysis of text."
LLM_USER = (
    "Classify the emotion conveyed by the speaker in the following sentence. "
    "Consider both explicit and implicit emotional meaning.\n"
    "Options: angry, happy, neutral, sad.\n"
    "Answer with exactly one word from the options.\n\n"
    "Sentence: {text}"
)

MODELS = {
    "hartmann_large": dict(kind="cls", repo="j-hartmann/emotion-english-roberta-large", act="softmax",
                           map={"angry": ["anger"], "happy": ["joy"], "neutral": ["neutral"], "sad": ["sadness"]}),
    "hartmann_distil": dict(kind="cls", repo="j-hartmann/emotion-english-distilroberta-base", act="softmax",
                            map={"angry": ["anger"], "happy": ["joy"], "neutral": ["neutral"], "sad": ["sadness"]}),
    # GoEmotions és multi-etiqueta (sigmoide): puntuació de classe = màxim de les etiquetes agrupades.
    "go_emotions": dict(kind="cls", repo="SamLowe/roberta-base-go_emotions", act="sigmoid",
                        map={"angry": ["anger", "annoyance", "disapproval"],
                             "happy": ["joy", "amusement", "excitement", "gratitude", "love",
                                       "optimism", "pride", "admiration", "relief"],
                             "neutral": ["neutral"],
                             "sad": ["sadness", "grief", "disappointment", "remorse"]}),
    "deberta_zeroshot": dict(kind="nli", repo="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
                             words={"angry": "angry", "happy": "happy", "neutral": "neutral", "sad": "sad"}),
    "qwen3_4b": dict(kind="llm", repo="Qwen/Qwen3-4B-Instruct-2507"),
}


# ---------------- Utilitats ----------------
def resolve_revision(repo: str, lock_path: Path) -> str:
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    if repo not in lock:
        lock[repo] = HfApi().model_info(repo).sha
        lock_path.write_text(json.dumps(lock, indent=2))
    return lock[repo]


def normalise(S: pd.DataFrame) -> pd.DataFrame:
    S = S[EMOS].clip(lower=EPS)
    return S.div(S.sum(axis=1), axis=0).add_prefix("p_")


def batches(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(i + bs, n))


# ---------------- Inferència per tipus de model ----------------
@torch.inference_mode()
def run_cls(cfg, rev, texts, bs, device):
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(cfg["repo"], revision=rev).to(device).eval()
    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    for emo, ls in cfg["map"].items():
        assert all(l in labels for l in ls), f"etiquetes inexistents a {cfg['repo']}: {ls} vs {labels}"
    out = []
    for sl in batches(len(texts), bs):
        enc = tok(texts[sl], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        lg = model(**enc).logits.float()
        out.append((torch.softmax(lg, -1) if cfg["act"] == "softmax" else torch.sigmoid(lg)).cpu().numpy())
    R = pd.DataFrame(np.vstack(out), columns=["raw_" + l for l in labels])
    S = pd.DataFrame({e: R[["raw_" + l for l in ls]].max(axis=1) for e, ls in cfg["map"].items()})
    return R, S, {"labels": labels, "map": cfg["map"], "activation": cfg["act"], "aggregation": "max"}


@torch.inference_mode()
def run_nli(cfg, rev, texts, bs, device):
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(cfg["repo"], revision=rev).to(device).eval()
    ent = [i for i, l in model.config.id2label.items() if l.lower() == "entailment"]
    assert len(ent) == 1, model.config.id2label
    S = {}
    for emo in EMOS:
        hyp = NLI_TEMPLATE.format(cfg["words"][emo])
        col = []
        for sl in batches(len(texts), bs):
            b = texts[sl]
            enc = tok(b, [hyp] * len(b), padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(device)
            col.append(torch.softmax(model(**enc).logits.float(), -1)[:, ent[0]].cpu().numpy())
        S[emo] = np.concatenate(col)
    S = pd.DataFrame(S)
    R = S.add_prefix("raw_entail_")
    return R, S, {"template": NLI_TEMPLATE, "words": cfg["words"], "score": "P(entailment) per hipòtesi"}


@torch.inference_mode()
def run_llm(cfg, rev, texts, bs, device):
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=rev)
    tok.padding_side = "left"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["repo"], revision=rev, torch_dtype=dtype).to(device).eval()
    ids = [tok.encode(e, add_special_tokens=False)[0] for e in EMOS]
    assert len(set(ids)) == 4, "el primer token de les 4 respostes ha de ser diferent"
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": LLM_USER.format(text=t)}],
        tokenize=False, add_generation_prompt=True) for t in texts]
    out = []
    for sl in batches(len(prompts), bs):
        enc = tok(prompts[sl], return_tensors="pt", padding=True).to(device)
        last = model(**enc).logits[:, -1, :].float()
        out.append(torch.log_softmax(last, -1)[:, ids].cpu().numpy())   # logprob completa de cada opció
    LP = np.vstack(out)
    R = pd.DataFrame(LP, columns=["raw_logprob_" + e for e in EMOS])
    S = pd.DataFrame(np.exp(LP), columns=EMOS)
    return R, S, {"system": LLM_SYSTEM, "user_template": LLM_USER, "decoding": "sense generació: logprob del "
                  "primer token de cada opció (determinista, equivalent a greedy restringit)",
                  "option_token_ids": dict(zip(EMOS, map(int, ids))), "dtype": str(dtype)}


RUNNERS = {"cls": run_cls, "nli": run_nli, "llm": run_llm}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=ROOT / "data/processed/emis_manifest.csv")
    ap.add_argument("--out-dir", default=ROOT / "outputs/text_baseline")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out = Path(a.out_dir)
    (out / "features").mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(a.manifest, dtype=str, usecols=["text", "text_id", "text_category", "text_emotion"])
    uniq = man.drop_duplicates("text").reset_index(drop=True)          # frase literal única
    assert man.groupby("text")["text_emotion"].nunique().max() == 1, "mateix text amb etiquetes diferents"
    texts = uniq["text"].tolist()                                      # ÚNIC input dels models
    print(f"{len(man)} àudios -> {len(texts)} textos únics | device={a.device} | transformers {transformers.__version__}")

    meta_path = out / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["_global"] = {"seed": SEED, "torch": torch.__version__, "transformers": transformers.__version__,
                       "python": platform.python_version(), "n_unique_texts": len(texts),
                       "manifest_sha256": hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
                       "input_column": "text (únic)", "classes": EMOS}

    for name in a.models:
        cfg = MODELS[name]
        rev = resolve_revision(cfg["repo"], out / "model_revisions.json")
        t0 = time.time()
        R, S, info = RUNNERS[cfg["kind"]](cfg, rev, texts, a.batch_size, a.device)
        df = pd.concat([uniq[["text", "text_category", "text_id"]], normalise(S), R], axis=1)
        df.to_csv(out / "features" / f"{name}.csv", index=False)
        meta[name] = {"repo": cfg["repo"], "revision": rev, "kind": cfg["kind"], "batch_size": a.batch_size,
                      "device": a.device, "seconds": round(time.time() - t0, 1), **info}
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  {name:18s} rev={rev[:8]}  {time.time() - t0:.0f} s")
        if a.device == "cuda":
            torch.cuda.empty_cache()
    print(f"\nFeatures: {out / 'features'} | meta: {meta_path}")


if __name__ == "__main__":
    main()
