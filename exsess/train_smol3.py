#!/usr/bin/env python
"""
Fine-tune HuggingFaceTB/SmolLM3-3B on a tiny slice of trl-lib/Capybara using TRL SFTTrainer + DeepSpeed ZeRO-3.

Quick start (single node, multiple GPUs):

  deepspeed train_sft_smol.py \
    --output_dir outputs/smol3b-capybara-mini \
    --deepspeed deepspeed_config.json \
    --max_steps 100 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5

The script intentionally uses a very small subset of the dataset to validate the end-to-end pipeline.
Increase --train_examples (and/or remove --max_steps) once things work.
"""

import argparse
import os
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, apply_chat_template
import math
import operator
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="swiss-ai/Apertus-8B-Instruct-2509")
    p.add_argument("--dataset_name", type=str, default="trl-lib/Capybara")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--train_examples", type=int, default=200, help="How many examples to use for a quick smoke test")
    p.add_argument("--output_dir", type=str, default="results/llama_ed")
    p.add_argument("--deepspeed", type=str, default='./deepspeed_config.json',
                        help="Path to DeepSpeed config JSON")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--attn_impl", type=str, default=None, choices=["flash_attention_2", "sdpa", "eager"],
        help="Override attention implementation; defaults to flash_attention_2 if available, else sdpa.")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", -1)),
                        help="Provided by DeepSpeed/torchrun.")
    return p.parse_known_args()

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

def freeze_bottom_fraction_of_layers(model: nn.Module, fraction: float = 0.75):
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

def main():
    args, _ = build_argparser()
    torch.backends.cuda.matmul.allow_tf32 = True

    # -------- Model & Tokenizer --------
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else None,
        "trust_remote_code": True,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # Important for training + gradient checkpointing
    model.config.use_cache = False
    # Enable gradient checkpointing to reduce memory while training the top 1/4
    model.gradient_checkpointing_enable()
    model.config.gradient_checkpointing = True

    # Freeze first 3/4 transformer layers
    freeze_bottom_fraction_of_layers(model, fraction=0.75)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name)
    if tokenizer.pad_token is None:
        # Many Llama/Qwen tokenizers don't define pad; set to eos for SFT convenience
        tokenizer.pad_token = tokenizer.eos_token

    # -------- Dataset (tiny slice for smoke test) --------
    ds = load_dataset(path="HuggingFaceTB/smoltalk", name="everyday-conversations")
    ds = ds.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    # -------- SFT Trainer --------
    print("Building SFTTrainer…")
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_steps=400,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=100,
        save_total_limit=2,
        save_steps=400,
        eval_steps=200,
        deepspeed=args.deepspeed,
        gradient_checkpointing=True,  # keep Trainer-aware as well
        packing=True,
        assistant_only_loss=True
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
    )

    trainer.train()

if __name__ == "__main__":
    main()

