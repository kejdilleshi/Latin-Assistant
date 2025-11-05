#!/usr/bin/env python
import argparse
import os
import json
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,  # NEW: only if you later want callbacks
    set_seed,
)
import operator
from trl import SFTTrainer, SFTConfig, apply_chat_template
from typing import Dict, List
import math
def read_jsonl(path: str) -> List[Dict]:
    allowed_tasks = {"translate_idiomatic", "transform","construct_tagging","translate_literal","morphosyntax","contrast_judge"}
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


# --- helper to find the decoder block stack across model families ---
def _get_decoder_layers_module(model: nn.Module):
    """
    Return (layers_module, attr_path_string).
    Tries common paths across HF architectures (Qwen/Llama/NeoX/etc.).
    """
    candidates = [
        "model.layers",            # Llama/Qwen2/Qwen3 style (e.g., Qwen2ForCausalLM.model.layers)
        "model.decoder.layers",    # some encoder-decoder or decoder wrappers
        "transformer.h",           # GPT-NeoX / GPT-J style
        "transformer.layers",      # some GPT styles
        "gpt_neox.layers",         # older NeoX wrappers
    ]
    for path in candidates:
        try:
            layers = operator.attrgetter(path)(model)
            if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
                return layers, path
        except Exception:
            pass
    raise RuntimeError(
        "Could not locate transformer block stack. "
        "Inspect the model architecture to adapt the layer path."
    )

def set_model_dropout(model, p=0.1):
    # set config fields if they exist
    for attr in ["attention_dropout", "hidden_dropout", "attn_dropout",
                 "hidden_dropout_prob", "attention_probs_dropout_prob"]:
        if hasattr(model.config, attr):
            setattr(model.config, attr, p)
    # also patch any Dropout modules already in the graph
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p

def freeze_bottom_fraction_of_layers(model: nn.Module, fraction: float = 0.80):
    """
    Freeze the bottom `fraction` of the transformer layers.
    Leaves embeddings and the remaining top layers trainable.
    """
    layers, path = _get_decoder_layers_module(model)
    n_total = len(layers)
    n_freeze = int(math.floor(n_total * fraction))
    # Ensure everything is trainable first, then freeze the bottom chunk
    model.requires_grad_(True)
    for i in range(n_freeze):
        layers[i].requires_grad_(False)

    # Optional: you can also freeze final layer norms in frozen blocks remain frozen by default above.
    # Print a small report
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[Freeze Report] Found layers at '{path}' with {n_total} blocks. "
        f"Froze bottom {n_freeze} ({fraction*100:.1f}%). "
        f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M."
    )

def build_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--dataset_name", type=str, default="everyday-conversations")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--output_dir", type=str, default="results/Qwen_latin_standart")
    p.add_argument("--deepspeed", type=str, default='./deepspeed_config.json',
                        help="Path to DeepSpeed config JSON")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", -1)),
                        help="Provided by DeepSpeed/torchrun.")
    return p.parse_known_args()


def setup_model(args):
    """Initialize and configure the model with gradient checkpointing and layer freezing."""
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else None,
        "trust_remote_code": True,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # Important for training + gradient checkpointing
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.config.gradient_checkpointing = True

    # Freeze bottom 50% of transformer layers
    freeze_bottom_fraction_of_layers(model, fraction=0.50)

    return model


def setup_tokenizer(model_name: str):
    """Initialize tokenizer and set padding token if needed."""
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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


def create_trainer(model, train_ds, val_ds, args):
    """Create and configure the SFTTrainer."""
    print("Building SFTTrainer…")
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        learning_rate=args.learning_rate,
        num_train_epochs=1,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        do_eval=True,
        eval_steps=50,
        eval_strategy="steps",
        deepspeed=args.deepspeed,
        gradient_checkpointing=True,
        packing=True,
        completion_only_loss=True,
        max_length=1024
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    return trainer


def main():
    args, _ = build_argparser()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = setup_model(args)
    tokenizer = setup_tokenizer(args.model_name)
    train_ds, test_ds, val_ds = load_and_split_datasets(args, tokenizer)
    trainer = create_trainer(model, train_ds, val_ds, args)

    trainer.train()


if __name__ == "__main__":
    main()
