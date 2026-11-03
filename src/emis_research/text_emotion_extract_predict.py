"""

PAS 1 de 2 — Extracció de probabilitats d'emoció a partir del text (sense fine-tuning).
 
Per a cada text únic d'EMIS, models preentrenats donen una probabilitat per a angry, happy,
neutral i sad. L'únic input dels models és el text; l'etiqueta només serveix per comprovar coherència.
  - Classificadors: hartmann_large, hartmann_distil, go_emotions, emoberta_large (etiquetes natives
    agrupades en les 4 d'EMIS).
  - Zero-shot NLI: deberta_zeroshot (P(entailment) de "The speaker feels <emoció>.").
  - LLM: qwen3_4b, qwen3_8b (probabilitat de la resposta, sense generar text).
 
Genera a outputs/text_baseline/:
  features/<model>.csv   Un per model: text, p_angry/p_happy/p_neutral/p_sad (sumen 1) i
                         puntuacions originals raw_*. ÉS L'ENTRADA DEL PAS 2.
  run_meta.json          Metadades de l'execució (versions, prompts, mapatges, temps).
  model_revisions.json   Commit SHA de cada model de Hugging Face, per reproduir-ho.
 
Dependències: data/processed/emis_manifest.csv (text, text_id, text_category, text_emotion);
torch, transformers, huggingface_hub, numpy, pandas; internet la primera vegada; GPU recomanada.
Següent pas: text_emotion_zero_shot.py.

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
EMOS = ["angry", "happy", "neutral", "sad"]   # ordre fix de les 4 classes d'EMIS (s'usa a tot arreu)
EPS = 1e-9                                    # evita probabilitats 0 abans de normalitzar

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

# Registre de models. `map` diu quines etiquetes NATIVES del model compten com cada emoció d'EMIS.
# Les emocions natives que no apareixen al mapatge (disgust, fear, surprise...) s'ignoren.
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
    # --- Candidatos afegits després de la primera ronda (escollits per literatura, no per resultats) ---
    # EmoBERTa-large: RoBERTa-large entrenat en MELD+IEMOCAP (conversa parlada, inclou neutral).
    # Aquí s'usa sense context de conversa (frase aïllada). Les etiquetes s'han de verificar amb l'assert.
    "emoberta_large": dict(kind="cls", repo="tae898/emoberta-large", act="softmax",
                           map={"angry": ["anger"], "happy": ["joy"], "neutral": ["neutral"], "sad": ["sadness"]}),
    # Qwen3-8B (híbrid thinking/non-thinking): es força enable_thinking=False; mateix prompt que qwen3_4b.
    "qwen3_8b": dict(kind="llm", repo="Qwen/Qwen3-8B", chat_kwargs={"enable_thinking": False}),
}


# ---------------- Utilitats ----------------
def resolve_revision(repo: str, lock_path: Path) -> str:
    """Retorna el commit SHA del checkpoint. La primera vegada el demana a Hugging Face i el desa a
    model_revisions.json; les següents el reutilitza. Així, tothom executa exactament la mateixa versió
    del model encara que l'autor el actualitzi després."""
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    if repo not in lock:
        lock[repo] = HfApi().model_info(repo).sha
        lock_path.write_text(json.dumps(lock, indent=2))
    return lock[repo]


def normalise(S: pd.DataFrame) -> pd.DataFrame:
    """Converteix les 4 puntuacions en probabilitats que sumen 1 per fila (columnes p_angry...p_sad).
    El clip evita dividir per 0 si una puntuació és exactament 0."""
    S = S[EMOS].clip(lower=EPS)
    return S.div(S.sum(axis=1), axis=0).add_prefix("p_")


def batches(n, bs):
    """Genera talls (slices) consecutius de mida bs per recórrer n elements per lots."""
    for i in range(0, n, bs):
        yield slice(i, min(i + bs, n))


# ---------------- Inferència per tipus de model ----------------
@torch.inference_mode()   # sense gradients: només inferència, estalvia memòria
def run_cls(cfg, rev, texts, bs, device):
    """Classificador d'emocions ja entrenat (Hartmann, GoEmotions, EmoBERTa).
    El model dona una probabilitat per a cadascuna de les SEVES etiquetes; nosaltres les agrupem
    segons cfg["map"] en les 4 d'EMIS."""
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=rev)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(cfg["repo"], revision=rev)
    except OSError:  # repo només amb pesos TensorFlow (requereix tensorflow instal·lat)
        model = AutoModelForSequenceClassification.from_pretrained(cfg["repo"], revision=rev, from_tf=True)
    model = model.to(device).eval()
    # Etiquetes natives del model, en l'ordre dels seus índexs de sortida.
    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    # Comprovació: si el mapatge nomena una etiqueta que el model no té, falla ara i no en silenci.
    for emo, ls in cfg["map"].items():
        assert all(l in labels for l in ls), f"etiquetes inexistents a {cfg['repo']}: {ls} vs {labels}"
    out = []
    for sl in batches(len(texts), bs):
        enc = tok(texts[sl], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        lg = model(**enc).logits.float()
        # softmax: classes excloents (suma 1). sigmoide: cada etiqueta independent (GoEmotions, multi-etiqueta).
        out.append((torch.softmax(lg, -1) if cfg["act"] == "softmax" else torch.sigmoid(lg)).cpu().numpy())
    R = pd.DataFrame(np.vstack(out), columns=["raw_" + l for l in labels])
    # Puntuació de cada emoció d'EMIS = màxim de les etiquetes natives que hi hem agrupat.
    S = pd.DataFrame({e: R[["raw_" + l for l in ls]].max(axis=1) for e, ls in cfg["map"].items()})
    return R, S, {"labels": labels, "map": cfg["map"], "activation": cfg["act"], "aggregation": "max"}


@torch.inference_mode()
def run_nli(cfg, rev, texts, bs, device):
    """Model d'inferència textual (NLI): decideix si un text IMPLICA una hipòtesi.
    Per cada text es construeixen 4 hipòtesis ("The speaker feels angry/happy/neutral/sad.") i es pren
    P(entailment) de cadascuna. El model no coneix les classes d'EMIS: només raona sobre si la frase
    implica la hipòtesi. Aquest model és binari (entailment / not_entailment); "neutral" aquí és
    una de les 4 emocions d'EMIS, no una classe del model."""
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(cfg["repo"], revision=rev).to(device).eval()
    # Índex de la classe "entailment" dins la sortida del model (ha d'haver-n'hi exactament una).
    ent = [i for i, l in model.config.id2label.items() if l.lower() == "entailment"]
    assert len(ent) == 1, model.config.id2label
    S = {}
    for emo in EMOS:                                   # una passada per hipòtesi (4 en total)
        hyp = NLI_TEMPLATE.format(cfg["words"][emo])
        col = []
        for sl in batches(len(texts), bs):
            b = texts[sl]
            # El tokenitzador rep parelles (premissa = text, hipòtesi = la mateixa per a tot el lot).
            enc = tok(b, [hyp] * len(b), padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(device)
            col.append(torch.softmax(model(**enc).logits.float(), -1)[:, ent[0]].cpu().numpy())
        S[emo] = np.concatenate(col)
    S = pd.DataFrame(S)
    R = S.add_prefix("raw_entail_")
    return R, S, {"template": NLI_TEMPLATE, "words": cfg["words"], "score": "P(entailment) per hipòtesi"}


@torch.inference_mode()
def run_llm(cfg, rev, texts, bs, device):
    """Puntuació principal = P(la PRIMERA PARAULA de la resposta és l'opció), sense importar què vingui després.

    P(primera paraula = "Sad") = P(tokens de "Sad" | prompt) * P(token següent NO continua la paraula)

    El segon factor evita comptar "Sadness"/"Sadly": un token següent és "frontera de paraula" si el seu
    text no comença per una lletra (espai, coma, punt, salt de línia, <|im_end|>, etc.). Així
    "Sad, because...", "Sad because..." i "Sad" compten com a sad. "Not sad" NO compta (primera paraula "Not").
    Es sumen les variants "sad" i "Sad". Després p_* es renormalitza entre les 4 opcions.

    Diagnòstic (mateixa passada, cost zero): raw_logprob_exact_* = P(paraula + <|im_end|>), la resposta
    exacta d'una sola paraula; option_mass = P(primera paraula ∈ 4 opcions) abans de renormalitzar.
    Limitació: només es considera la tokenització canònica de la paraula (p. ex. no "S"+"ad").

    NO es genera text: es fa una passada del model per cada (text, opció, variant) amb la resposta
    candidata enganxada al prompt, i es llegeixen les probabilitats que el model assigna a aquells tokens.
    """
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=rev)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["repo"], revision=rev, torch_dtype=dtype).to(device).eval()
    end_id = tok.convert_tokens_to_ids("<|im_end|>")          # token de "final de torn" de Qwen
    assert end_id is not None and end_id != tok.unk_token_id, "no es troba <|im_end|>"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else end_id

    # --- 1) Respostes candidates: cada emoció en minúscula i Capitalitzada, convertides a tokens ---
    variants = {e: [e, e.capitalize()] for e in EMOS}
    cont = {e: [tok.encode(v, add_special_tokens=False) for v in variants[e]] for e in EMOS}  # sense <|im_end|>
    tokenization = {e: [[tok.decode([t]) for t in c] for c in cont[e]] for e in EMOS}         # per inspecció
    print("    tokenització de les opcions:", tokenization)

    # --- 2) Màscara de "frontera de paraula" sobre TOT el vocabulari ---
    # Un token és frontera si el seu text no comença per una lletra: espai, puntuació, salt de línia,
    # <|im_end|>... Un token com "ness" o "ly" NO és frontera (continuaria la paraula: Sad+ness).
    V = model.get_output_embeddings().weight.shape[0]                 # mida real dels logits
    boundary = torch.zeros(V, dtype=torch.bool)
    for t in range(min(len(tok), V)):                                 # token = frontera si no comença per lletra
        d = tok.decode([t])
        boundary[t] = len(d) > 0 and not d[0].isalpha()
    boundary = boundary.to(device)
    print(f"    tokens frontera de paraula: {int(boundary.sum())} de {V}")

    # --- 3) Prompt de cada text, amb la plantilla de xat del model, convertit a ids ---
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": LLM_USER.format(text=t)}],
        tokenize=False, add_generation_prompt=True, **cfg.get("chat_kwargs", {})) for t in texts]
    p_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]

    # --- 4) Una "tasca" per cada (text, emoció, variant): prompt + paraula candidata ---
    items = [(i, e, v, p_ids[i], c) for i in range(len(texts)) for e in EMOS
             for v, c in enumerate(cont[e])]
    sc_first, sc_exact = np.zeros(len(items)), np.zeros(len(items))
    for sl in batches(len(items), bs):
        chunk = items[sl]
        seqs = [it[3] + it[4] for it in chunk]            # seqüència = prompt + paraula candidata
        T = max(map(len, seqs))
        ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)    # padding a la dreta
        att = torch.zeros((len(seqs), T), dtype=torch.long)           # 1 = token real, 0 = padding
        for r, q in enumerate(seqs):
            ids[r, :len(q)] = torch.tensor(q)
            att[r, :len(q)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        for r, it in enumerate(chunk):
            L, c = len(it[3]), it[4]                       # L = longitud del prompt, c = tokens de la paraula
            # El logit a la posició p prediu el token p+1. Per tant, els tokens de la paraula (posicions
            # L..L+len(c)-1) es prediuen amb les posicions L-1..L+len(c)-2, i la posició L+len(c)-1 dona
            # la distribució del token SEGÜENT a la paraula. En total, len(c)+1 posicions.
            lp = torch.log_softmax(logits[r, L - 1:L + len(c)].float(), dim=-1)
            prefix = lp[:-1].gather(1, torch.tensor(c, device=lp.device)[:, None]).sum()  # log P(paraula)
            nxt = lp[-1]                                                                  # token següent
            # log P(paraula) + log P(token següent és frontera) = primera paraula completa, qualsevol cosa després
            sc_first[sl.start + r] = (prefix + torch.logsumexp(nxt[boundary], dim=0)).item()
            # Diagnòstic: log P(paraula) + log P(<|im_end|>) = resposta exacta d'una sola paraula
            sc_exact[sl.start + r] = (prefix + nxt[end_id]).item()

    # --- 5) Agrupar per (text, emoció): sumar probabilitats de les variants (logaddexp = suma en log) ---
    LP = np.full((len(texts), len(EMOS)), -np.inf)
    LPx = np.full((len(texts), len(EMOS)), -np.inf)
    for (i, e, v, _, _), a1, a2 in zip(items, sc_first, sc_exact):
        j = EMOS.index(e)
        LP[i, j] = np.logaddexp(LP[i, j], a1)                       # suma de variants
        LPx[i, j] = np.logaddexp(LPx[i, j], a2)
    # option_mass = probabilitat que la resposta sigui UNA de les 4 opcions (abans de renormalitzar).
    # Si és baixa, el model vol dir una altra cosa i la renormalització amaga aquest fet.
    mass, mass_x = np.exp(LP).sum(axis=1), np.exp(LPx).sum(axis=1)
    R = pd.DataFrame(LP, columns=["raw_logprob_firstword_" + e for e in EMOS])
    R = pd.concat([R, pd.DataFrame(LPx, columns=["raw_logprob_exact_" + e for e in EMOS])], axis=1)
    R["option_mass"] = mass
    R["option_mass_exact"] = mass_x
    S = pd.DataFrame(np.exp(LP), columns=EMOS)
    return R, S, {"system": LLM_SYSTEM, "user_template": LLM_USER,
                  "scoring": "P(primera paraula = opció) = P(tokens de la paraula | prompt) * P(token següent és "
                             "frontera de paraula); logsumexp sobre variants minúscula/Capitalitzada; "
                             "p_* renormalitzat entre les 4 opcions",
                  "boundary_rule": "token el text del qual no comença per una lletra",
                  "variants": variants, "option_tokenization": tokenization, "dtype": str(dtype),
                  "chat_kwargs": cfg.get("chat_kwargs", {}),
                  "option_mass_mean": float(mass.mean()), "option_mass_min": float(mass.min()),
                  "option_mass_exact_mean": float(mass_x.mean()), "option_mass_exact_min": float(mass_x.min())}


# Cada tipus de model té la seva funció d'inferència (totes retornen R, S, info).
RUNNERS = {"cls": run_cls, "nli": run_nli, "llm": run_llm}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=ROOT / "data/processed/emis_manifest.csv")
    ap.add_argument("--out-dir", default=ROOT / "outputs/text_baseline")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    # Llavors fixes (la inferència ja és determinista, però ho deixem registrat).
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out = Path(a.out_dir)
    (out / "features").mkdir(parents=True, exist_ok=True)

    # PAS 1: textos únics. Es llegeixen text_id/text_category/text_emotion només per traçabilitat i per
    # comprovar coherència; NO entren a cap model.
    man = pd.read_csv(a.manifest, dtype=str, usecols=["text", "text_id", "text_category", "text_emotion"])
    uniq = man.drop_duplicates("text").reset_index(drop=True)          # frase literal única
    assert man.groupby("text")["text_emotion"].nunique().max() == 1, "mateix text amb etiquetes diferents"
    texts = uniq["text"].tolist()                                      # ÚNIC input dels models
    print(f"{len(man)} àudios -> {len(texts)} textos únics | device={a.device} | transformers {transformers.__version__}")

    # PAS 2: metadades globals de l'execució (si ja existeix run_meta.json, es conserva i s'actualitza).
    meta_path = out / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["_global"] = {"seed": SEED, "torch": torch.__version__, "transformers": transformers.__version__,
                       "python": platform.python_version(), "n_unique_texts": len(texts),
                       "manifest_sha256": hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
                       "input_column": "text (únic)", "classes": EMOS}

    # PAS 3: per cada model → revisió fixada → inferència → normalitzar → desar CSV + metadades.
    for name in a.models:
        cfg = MODELS[name]
        rev = resolve_revision(cfg["repo"], out / "model_revisions.json")
        t0 = time.time()
        R, S, info = RUNNERS[cfg["kind"]](cfg, rev, texts, a.batch_size, a.device)
        # CSV final: traçabilitat + p_* (probabilitats normalitzades) + sortides natives raw_*.
        df = pd.concat([uniq[["text", "text_category", "text_id"]], normalise(S), R], axis=1)
        df.to_csv(out / "features" / f"{name}.csv", index=False)
        meta[name] = {"repo": cfg["repo"], "revision": rev, "kind": cfg["kind"], "batch_size": a.batch_size,
                      "device": a.device, "seconds": round(time.time() - t0, 1), **info}
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  {name:18s} rev={rev[:8]}  {time.time() - t0:.0f} s")
        if a.device == "cuda":
            torch.cuda.empty_cache()                                   # allibera GPU abans del model següent
    print(f"\nFeatures: {out / 'features'} | meta: {meta_path}")


if __name__ == "__main__":
    main()