#!/usr/bin/env python
"""Main training script for supervised fine-tuning with DeepSpeed."""

import argparse
import os
import torch
from transformers import set_seed
import wandb
from model_utils import setup_model, setup_tokenizer
from data_utils import load_and_split_datasets
from trainer_utils import create_trainer


def build_argparser() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM3-3B")
    p.add_argument("--dataset_name", type=str, default="local")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--output_dir", type=str, default="results/SmolLM3_Latin_sft_packing_bs4_lr1e-6_ep1_")
    p.add_argument("--deepspeed", type=str, default='./deepspeed_config.json',
                   help="Path to DeepSpeed config JSON")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_rank", type=int,
                   default=int(os.environ.get("LOCAL_RANK", -1)),
                   help="Provided by DeepSpeed/torchrun.")

    # Weights & Biases arguments
    p.add_argument("--wandb_project", type=str, default="Train-sft",
                   help="WandB project name")
    p.add_argument("--wandb_entity", type=str, default="kejdi-lleshi-university-of-lausanne",
                   help="WandB entity/username")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="WandB run name (defaults to output_dir basename)")
    p.add_argument("--use_wandb", action="store_true", default=True,
                   help="Enable WandB logging")

    # Training configuration
    p.add_argument("--packing", action="store_true", default=False,
                   help="Enable packing for SFT training")
    p.add_argument("--local_data_path", type=str, default="data/sft_loredana_plus/sft_items.jsonl",
                   help="Path to local JSONL data file (used when dataset_name='local')")


    return p.parse_known_args()


def main():
    """Main training orchestration."""
    args, _ = build_argparser()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Set up Weights & Biases environment variables
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name

    # Setup model and tokenizer
    model = setup_model(args)
    tokenizer = setup_tokenizer(args.model_name)

    # Load and preprocess datasets
    train_ds, test_ds, val_ds = load_and_split_datasets(args, tokenizer)

    # Create trainer and start training
    trainer = create_trainer(model, train_ds, val_ds, args)
    trainer.train()

    # Save the final model and tokenizer
    final_output_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_output_dir)
    print(f"Model and tokenizer saved to {final_output_dir}")


if __name__ == "__main__":
    main()
