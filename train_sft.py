#!/usr/bin/env python
"""Main training script for supervised fine-tuning with DeepSpeed."""

import argparse
import os
import torch
from transformers import set_seed

from model_utils import setup_model, setup_tokenizer
from data_utils import load_and_split_datasets
from trainer_utils import create_trainer


def build_argparser() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
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


def main():
    """Main training orchestration."""
    args, _ = build_argparser()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Setup model and tokenizer
    model = setup_model(args)
    tokenizer = setup_tokenizer(args.model_name)

    # Load and preprocess datasets
    train_ds, test_ds, val_ds = load_and_split_datasets(args, tokenizer)

    # Create trainer and start training
    trainer = create_trainer(model, train_ds, val_ds, args)
    trainer.train()


if __name__ == "__main__":
    main()
