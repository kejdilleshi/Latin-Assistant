#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast
import os
from typing import List

import torch
import numpy as np
from transformers import AutoTokenizer


def load_ids(source) -> torch.Tensor:
    """
    Load input_ids from:
      - torch.Tensor / np.ndarray / (list|tuple) in memory
      - .pt (torch.save) file
      - .npy (numpy) file
      - Python literal string (e.g., "tensor([[1,2,3]])" or "[[1,2,3]]")
    Returns a 2D LongTensor of shape (batch, seq_len).
    """
    # --- In-memory fast paths ---
    if isinstance(source, torch.Tensor):
        data = source
    elif isinstance(source, np.ndarray):
        data = source
    elif isinstance(source, (list, tuple)):
        data = np.array(source)
    # --- String: path or literal ---
    elif isinstance(source, str):
        if os.path.isfile(source):
            if source.endswith(".pt"):
                data = torch.load(source, map_location="cpu")
            elif source.endswith(".npy"):
                data = np.load(source, allow_pickle=False)
            else:
                raise ValueError("Unrecognized file extension. Use .pt or .npy, or pass a Python literal.")
        else:
            try:
                lit = ast.literal_eval(source)
            except Exception as e:
                raise ValueError(f"Could not parse input_ids literal: {e}")
            if isinstance(lit, torch.Tensor):
                data = lit
            elif isinstance(lit, (list, tuple)):
                data = np.array(lit)
            elif isinstance(lit, np.ndarray):
                data = lit
            else:
                raise ValueError("Literal must be a tensor/array/list/tuple.")
    else:
        raise ValueError(f"Unsupported input type: {type(source)}")

    # Normalize to LongTensor on CPU, shape (batch, seq)
    if isinstance(data, np.ndarray):
        t = torch.as_tensor(data, dtype=torch.long)
    elif isinstance(data, torch.Tensor):
        t = data.to(dtype=torch.long, device="cpu")
    else:
        raise ValueError("Unsupported data type after loading.")

    if t.ndim == 1:
        t = t.unsqueeze(0)  # (seq,) -> (1, seq)
    if t.ndim != 2:
        raise ValueError(f"input_ids must be 1D or 2D; got shape {tuple(t.shape)}")

    return t


def decode_batch(tokenizer: AutoTokenizer, batch_ids: torch.Tensor, skip_special_tokens: bool = False) -> List[str]:
    texts = tokenizer.batch_decode(
        batch_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False
    )
    return [t.strip() for t in texts]


def main():
    ap = argparse.ArgumentParser(description="Decode token IDs to text using a local tokenizer.")
    ap.add_argument("--model_dir", type=str, default="results/mistral_sft_full",
                    help="Path to folder with tokenizer files (tokenizer.json, etc.).")
    ap.add_argument("--keep_special", default=True, action="store_true", help="Keep special tokens in the decoded text.")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, local_files_only=True)

    # Not strictly needed for decode, but avoids warnings elsewhere
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Your in-memory IDs (1D is fine; loader will add batch dim)
    input_ids = torch.tensor([    3,  5586, 29515,  6773, 14497,  3858,  2317,  2140, 29491, 14109,
          1040,  5482,  1392,  1072,  1909,  1040,  2886,  1065,  1040,  1567,
         29491,  1098, 29499,  1294,  1082,  1586,  1771,  1472,  2145,  1032,
          8591, 25303,  1063,  2492,  1149,  1032,  1067, 27477, 28807,  1178,
          2520,  1151,  1133, 29499,  1294,  1082,  1586,  1771,  1472,  2145,
          1032,  8591, 25303,  1063,  2492,  1149,  1032,  1067, 27477,  9597,
          1153,     4, 10671,  3891, 29515,  1098, 29491,  8187,  1065,  1133,
         29515, 13401, 29501, 29523,  1046,  2498, 29491,     2], dtype=torch.long)
    

    ids = load_ids(input_ids)
    texts = decode_batch(tokenizer, ids, skip_special_tokens=not args.keep_special)

    for i, t in enumerate(texts):
        print(f"[{i}] {t}")


if __name__ == "__main__":
    main()
