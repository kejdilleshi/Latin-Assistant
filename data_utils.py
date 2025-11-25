"""Data loading and preprocessing utilities for SFT training."""

import json
from typing import Dict, List
from datasets import Dataset, load_dataset


def read_jsonl(path: str) -> List[Dict]:
    """Load and filter JSONL data by allowed tasks."""
    allowed_tasks = {
        "translate_idiomatic",
        "morphosyntax",
        "transform"
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

            # # Keep only allowed tasks; skip if missing or not allowed
            # task = obj.get("task")
            # if task not in allowed_tasks:
            #     continue

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
    """Load data from local JSONL or HuggingFace, split into train/test/val, and preprocess."""

    if args.dataset_name == "local":
        # Load local JSONL data
        local_path = getattr(args, 'local_data_path', 'data/sft_loredana_plus/sft_items.jsonl')
        print(f"Loading local dataset from JSONL file: {local_path}")
        all_items = read_jsonl(local_path)
        full_ds = Dataset.from_list(all_items)

        # Split: 80% train, 20% remainder
        split_80_20 = full_ds.train_test_split(test_size=0.10, seed=args.seed)
        train_ds = split_80_20["train"]
        rem_ds = split_80_20["test"]

        # From 20% remainder: 75% test (15% overall), 25% val (5% overall)
        rem_split = rem_ds.train_test_split(test_size=0.5, seed=args.seed)
        test_ds = rem_split["train"]
        val_ds = rem_split["test"]

        print(f"Split sizes -> train: {len(train_ds)}, test: {len(test_ds)}, val: {len(val_ds)}")

        # Preprocess all splits - only convert to messages format
        # SFTTrainer will handle tokenization internally
        train_ds = train_ds.map(preprocess_chat_format, remove_columns=["prompt", "target"])
        test_ds = test_ds.map(preprocess_chat_format, remove_columns=["prompt", "target"])
        val_ds = val_ds.map(preprocess_chat_format, remove_columns=["prompt", "target"])

    else:
        # Load dataset from HuggingFace
        print(f"Loading dataset from HuggingFace: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name, split=args.split)

        # Check if dataset already has predefined splits
        try:
            # Try to load predefined splits if they exist
            train_ds = load_dataset(args.dataset_name, split="train")
            test_ds = load_dataset(args.dataset_name, split="test")

            # Create validation split from train if no validation split exists
            try:
                val_ds = load_dataset(args.dataset_name, split="validation")
            except:
                # Split train to create validation set: 95% train, 5% val
                split_train_val = train_ds.train_test_split(test_size=0.05, seed=args.seed)
                train_ds = split_train_val["train"]
                val_ds = split_train_val["test"]

            print(f"Split sizes -> train: {len(train_ds)}, test: {len(test_ds)}, val: {len(val_ds)}")
        except:
            # No predefined splits, create our own
            print("No predefined splits found, creating splits...")
            # Split: 80% train, 20% remainder
            split_80_20 = dataset.train_test_split(test_size=0.20, seed=args.seed)
            train_ds = split_80_20["train"]
            rem_ds = split_80_20["test"]

            # From 20% remainder: 75% test (15% overall), 25% val (5% overall)
            rem_split = rem_ds.train_test_split(test_size=0.25, seed=args.seed)
            test_ds = rem_split["train"]
            val_ds = rem_split["test"]

            print(f"Split sizes -> train: {len(train_ds)}, test: {len(test_ds)}, val: {len(val_ds)}")

        # Preprocess HuggingFace datasets
        # Check if dataset already has 'messages' field (common for chat datasets)

        print("Preprocessing datasets, the dataset is in chat format:", train_ds.column_names)
        if "messages" in train_ds.column_names:
            # Dataset already in chat format, SFTTrainer will handle tokenization
            pass
        elif "prompt" in train_ds.column_names and "target" in train_ds.column_names:
            # Dataset has prompt/target format, convert to messages
            train_ds = train_ds.map(preprocess_completion_format, remove_columns=["prompt", "target"])
            test_ds = test_ds.map(preprocess_completion_format, remove_columns=["prompt", "target"])
            val_ds = val_ds.map(preprocess_completion_format, remove_columns=["prompt", "target"])
        else:
            # For other formats, try to convert to messages if possible
            print("Warning: Dataset format not recognized. SFTTrainer may have issues.")

    return train_ds, test_ds, val_ds
