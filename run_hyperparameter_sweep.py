#!/usr/bin/env python
"""
Hyperparameter sweep script for SFT training.

Runs multiple training configurations with different combinations of:
- Learning rates: 1e-6, 5e-6, 1e-5
- Batch sizes: 2, 4
- Epochs: 1
- Packing: False (can be extended to [False, True] in future)

Total: 6 runs (3 lr × 2 bs × 1 epoch × 1 packing)
"""

import os
import subprocess
import sys
from itertools import product
from datetime import datetime

# Define hyperparameter grid
LEARNING_RATES = [1e-5,5e-6]
BATCH_SIZES = [1, 2]
EPOCHS = [1,2]
PACKING = [True] 
# Common training arguments
BASE_ARGS = {
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "dataset_name": "local",
    "split": "train",
    "deepspeed": "./deepspeed_config.json",
    "logging_steps": 20,
    "save_steps": 40,
    "save_total_limit": 1,
    "seed": 42,
    "wandb_project": "Train-sft-sweep",
    "wandb_entity": "kejdi-lleshi-university-of-lausanne",
    "use_wandb": True,
}

def run_training_config(lr, bs, ep, packing, run_number, total_runs):
    """
    Run a single training configuration.

    Args:
        lr: Learning rate
        bs: Batch size
        ep: Number of epochs
        packing: Whether to use packing
        run_number: Current run number (1-indexed)
        total_runs: Total number of runs
    """
    # Create output directory name
    pack_str = "pack" if packing else "nopack"
    output_dir = f"/scratch/klleshi/Latin-chatbot/data/Llama_sweep_lr{lr:.0e}_bs{bs}_ep{ep}_{pack_str}_masked"
    wandb_run_name = f"llama_lr{lr:.0e}_bs{bs}_ep{ep}_{pack_str}"

    print("\n" + "="*80)
    print(f"RUN {run_number}/{total_runs}")
    print(f"Configuration: lr={lr:.0e}, batch_size={bs}, epochs={ep}, packing={packing}")
    print(f"Output directory: {output_dir}")
    print(f"WandB run name: {wandb_run_name}")
    print("="*80 + "\n")

    # Build command
    cmd = [
        "deepspeed",
        "--launcher", "slurm",
        "--num_nodes", str(os.environ.get("SLURM_JOB_NUM_NODES", "1")),
        "--hostfile", "hostfile",
        "--venv_script", "/work/CTR/CI/DCSR/rfabbret/llm/Kejdi/DeepSpeed/.venv/bin/activate",
        "train_sft.py",
        "--output_dir", output_dir,
        "--per_device_train_batch_size", str(bs),
        "--num_train_epochs", str(ep),
        "--learning_rate", str(lr),
        "--wandb_run_name", wandb_run_name,
    ]

    # Add packing flag if enabled
    if packing:
        cmd.append("--packing")

    # Add base arguments
    for key, value in BASE_ARGS.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    # Run the training
    print(f"Executing: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Run {run_number}/{total_runs} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Run {run_number}/{total_runs} failed with error code {e.returncode}")
        print(f"Error: {e}")
        return False

def main():
    """Run hyperparameter sweep."""
    # Generate all combinations
    configs = list(product(LEARNING_RATES, BATCH_SIZES, EPOCHS, PACKING))
    total_runs = len(configs)

    print("\n" + "="*80)
    print("HYPERPARAMETER SWEEP")
    print("="*80)
    print(f"Total configurations: {total_runs}")
    print(f"Learning rates: {LEARNING_RATES}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Epochs: {EPOCHS}")
    print(f"Packing: {PACKING}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    # Track results
    successful_runs = 0
    failed_runs = 0
    results = []

    # Run each configuration
    for i, (lr, bs, ep, packing) in enumerate(configs, 1):
        success = run_training_config(lr, bs, ep, packing, i, total_runs)

        results.append({
            "run": i,
            "lr": lr,
            "bs": bs,
            "ep": ep,
            "packing": packing,
            "success": success
        })

        if success:
            successful_runs += 1
        else:
            failed_runs += 1

    # Print training summary
    print("\n" + "="*80)
    print("TRAINING SWEEP SUMMARY")
    print("="*80)
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nDetailed results:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} Run {r['run']}: lr={r['lr']:.0e}, bs={r['bs']}, ep={r['ep']}, packing={r['packing']}")
    print("="*80 + "\n")

    # Exit with error if any training runs failed
    if failed_runs > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
