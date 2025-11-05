"""Data loading and preprocessing utilities for SFT training."""

import json
from typing import Dict, List
from datasets import Dataset
from trl import apply_chat_template


def read_jsonl(path: str) -> List[Dict]:
    """Load and filter JSONL data by allowed tasks."""
    allowed_tasks = {
        "translate_idiomatic",
        "transform",
        "construct_tagging",
        "translate_literal",
        "morphosyntax",
        "contrast_judge"
    }
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

            # Require prompt/target fields to be present
            if "prompt" not in obj or "target" not in obj:
                raise ValueError(f"Line {i} missing 'prompt' or 'target' field.")

            # Keep only allowed tasks; skip if missing or not allowed
            task = obj.get("task")
            if task not in allowed_tasks:
                continue

            prompt = str(obj["prompt"]).strip()
            target = str(obj["target"]).strip()
            if prompt == "" or target == "":
                continue  # skip empty items

            items.append({
                "prompt": prompt,
                "target": target,
            })

    if not items:
        raise ValueError("No valid items found in JSONL.")
    return items


def preprocess_completion_format(example):
    """Preprocess example into prompt/completion format for SFT."""
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "completion": [
            {"role": "assistant", "content": example['target']}
        ],
    }


def preprocess_chat_format(example):
    """Preprocess example into messages format for SFT."""
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example['target']}
        ],
    }


def load_and_split_datasets(args, tokenizer):
    """Load JSONL data, split into train/test/val, and preprocess."""
    # Load data
    all_items = read_jsonl("data/out_sft+/sft_items.jsonl")
    full_ds = Dataset.from_list(all_items)

    # Split: 80% train, 20% remainder
    split_80_20 = full_ds.train_test_split(test_size=0.20, seed=args.seed)
    train_ds = split_80_20["train"]
    rem_ds = split_80_20["test"]

    # From 20% remainder: 75% test (15% overall), 25% val (5% overall)
    rem_split = rem_ds.train_test_split(test_size=0.25, seed=args.seed)
    test_ds = rem_split["train"]
    val_ds = rem_split["test"]

    print(f"Split sizes -> train: {len(train_ds)}, test: {len(test_ds)}, val: {len(val_ds)}")

    # Preprocess all splits
    train_ds = train_ds.map(preprocess_completion_format, remove_columns=["prompt", "target"])
    train_ds = train_ds.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    test_ds = test_ds.map(preprocess_completion_format, remove_columns=["prompt", "target"])
    test_ds = test_ds.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    val_ds = val_ds.map(preprocess_completion_format, remove_columns=["prompt", "target"])
    val_ds = val_ds.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    return train_ds, test_ds, val_ds
