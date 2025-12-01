#!/usr/bin/env python
# inspect_token_lengths.py
import argparse, json, math, os, statistics
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer


# ---- keep this aligned with your training data filter ----
def read_jsonl(path: str) -> List[Dict]:
    allowed_tasks = {"translate_idiomatic", "transform","construct_tagging",
                     "translate_literal","morphosyntax","contrast_judge"}
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}")

            if "prompt" not in obj or "target" not in obj:
                raise ValueError(f"Line {i} missing 'prompt' or 'target' field.")

            task = obj.get("task")
            if task not in allowed_tasks:
                continue

            prompt = str(obj["prompt"]).strip()
            target = str(obj["target"]).strip()
            if prompt == "" or target == "":
                continue

            items.append({"prompt": prompt, "target": target})
    if not items:
        raise ValueError("No valid items found in JSONL.")
    return items


def preprocess_function(example):
    # Matches your code: one user turn, one assistant turn
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "completion": [{"role": "assistant", "content": example["target"]}],
    }


def messages_from_example(pp_example: Dict) -> List[Dict]:
    # Combine into the chat format expected by tokenizer.apply_chat_template
    # Example:
    # [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
    return pp_example["prompt"] + pp_example["completion"]


def count_tokens_with_chat_template(
    tokenizer: AutoTokenizer,
    messages: List[Dict],
    add_generation_prompt: bool = False
) -> int:
    """
    Prefer tokenizer.apply_chat_template(..., tokenize=True).
    Fall back to a simple join if the tokenizer has no chat template.
    """
    # Some tokenizers (esp. Instruct/chat models) expose a chat template.
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_tensors=None,
            )
            # Some tokenizers return a list[int], some a dict—normalize:
            if isinstance(token_ids, dict):  # very rare, but just in case
                token_ids = token_ids["input_ids"]
            return len(token_ids)
        except Exception:
            pass  # fall back below

    # Fallback: naive concatenation with role tags
    text = ""
    for m in messages:
        role = m.get("role", "user")
        text += f"<|{role}|>\n{m.get('content','')}\n"
    token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    return len(token_ids)


def next_multiple_of_8(n: int) -> int:
    return int(math.ceil(n / 8.0) * 8)


def suggest_max_length(
    lengths: List[int],
    model_limit: int,
    pct: float
) -> Tuple[int, float]:
    """
    Suggest a max_length as min(model_limit, ceil(pctile)->/8).
    Returns (suggested, truncation_rate_if_used).
    """
    pctile = float(np.percentile(lengths, pct))
    candidate = next_multiple_of_8(math.ceil(pctile))
    suggested = min(candidate, model_limit if model_limit and model_limit > 0 else candidate)
    trunc_rate = sum(1 for L in lengths if L > suggested) / len(lengths)
    return suggested, trunc_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="Percentile target for suggested max_length (e.g., 99.5).")
    ap.add_argument("--sample_limit", type=int, default=0,
                    help="If >0, only analyze the first N examples.")
    ap.add_argument("--out_csv", type=str, default="token_lengths.csv")
    ap.add_argument("--out_hist", type=str, default="token_lengths_hist.png",
                    help="If provided, saves a histogram PNG (requires matplotlib).")
    args = ap.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Get the model's hard cap if available
    # (tokenizer.model_max_length can be a sentinel like 1e30 for some models)
    model_limit = None
    try:
        # Prefer a "real" smallish integer; ignore huge sentinels
        mml = int(getattr(tokenizer, "model_max_length", 0))
        model_limit = mml if (mml and mml < 10_000_000) else None
    except Exception:
        model_limit = None

    # Load + preprocess to match your pipeline
    raw_items = read_jsonl(args.data_path)
    if args.sample_limit and args.sample_limit > 0:
        raw_items = raw_items[:args.sample_limit]
    prepped = [preprocess_function(x) for x in raw_items]

    # Count tokens per example as fed to SFT (no generation prompt)
    lengths = []
    for ex in tqdm(prepped, desc="Tokenizing examples"):
        msgs = messages_from_example(ex)
        n_tok = count_tokens_with_chat_template(tokenizer, msgs, add_generation_prompt=False)
        lengths.append(n_tok)

    # Save per-example lengths to CSV
    try:
        import csv
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "num_tokens"])
            for i, n in enumerate(lengths):
                w.writerow([i, n])
        print(f"[OK] Wrote per-example lengths to: {os.path.abspath(args.out_csv)}")
    except Exception as e:
        print(f"[WARN] Could not write CSV: {e}")

    # Optional histogram
    if args.out_hist:
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist(lengths, bins=50)
            plt.title("Token length distribution")
            plt.xlabel("Tokens")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(args.out_hist, dpi=160)
            print(f"[OK] Saved histogram to: {os.path.abspath(args.out_hist)}")
        except Exception as e:
            print(f"[WARN] Could not save histogram: {e}")

    # Stats
    L = np.array(lengths, dtype=np.int32)
    p50 = np.percentile(L, 50)
    p90 = np.percentile(L, 90)
    p95 = np.percentile(L, 95)
    p99 = np.percentile(L, 99)
    p995 = np.percentile(L, 99.5)
    Lmin, Lmax = int(L.min()), int(L.max())
    meanL = float(np.mean(L))
    stdL = float(np.std(L))

    # Suggestion at chosen percentile
    suggested, trunc_rate = suggest_max_length(lengths, model_limit, args.percentile)

    # Estimate padding waste if you pad every sequence to 'suggested'
    total_pad = int(sum(max(0, suggested - l) for l in lengths))
    total_tokens = int(sum(lengths))
    pad_ratio = total_pad / (total_tokens + total_pad) if (total_tokens + total_pad) > 0 else 0.0

    # Report
    print("\n=== Token Length Summary ===")
    print(f"Examples analyzed:   {len(lengths)}")
    if model_limit:
        print(f"Model limit:         {model_limit} tokens (from tokenizer.model_max_length)")
    else:
        print("Model limit:         unknown / not enforced by tokenizer")

    print(f"Min / Max:           {Lmin} / {Lmax}")
    print(f"Mean ± Std:          {meanL:.1f} ± {stdL:.1f}")
    print(f"p50 / p90 / p95:     {int(p50)} / {int(p90)} / {int(p95)}")
    print(f"p99 / p99.5:         {int(p99)} / {int(p995)}")

    print("\n=== Suggested SFT max_length ===")
    print(f"Target percentile:   p{args.percentile}")
    print(f"Suggested max_length: {suggested} "
          f"(multiple of 8; {'capped by model limit' if model_limit and suggested>=model_limit else 'not capped'})")
    print(f"Truncation rate @ suggested: {trunc_rate*100:.2f}% of examples would be truncated.")
    print(f"Padding share if always pad-to-{suggested}: {pad_ratio*100:.2f}% "
          f"(packing reduces this in practice).")

    # Also provide a few alternatives to eyeball tradeoffs
    for p in (98.0, 99.0, 99.5, 99.9):
        alt_len, alt_trunc = suggest_max_length(lengths, model_limit, p)
        print(f"  - If p{p:>4}: max_length={alt_len}, trunc={alt_trunc*100:.2f}%")

    print("\nTip:")
    print(" • Start with the suggested value. If truncation >~1–2% and those are important, bump it a notch.")
    print(" • With TRL packing=True, prefer a length that fits GPU memory comfortably over chasing the longest tail.")
    print(" • If you see frequent truncation in logs, rerun this with a higher percentile.")


if __name__ == "__main__":
    main()
