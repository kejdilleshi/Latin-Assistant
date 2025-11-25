#!/usr/bin/env python
"""
Benchmark all models from a hyperparameter sweep.

This script finds all trained models in sweep_* directories,
runs benchmarks on each model, and saves results in their respective folders.
"""

import argparse
import json
import os
import re
from pathlib import Path
from datetime import datetime
import torch
import wandb
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from run_benchmark import run_single_benchmark


def find_sweep_models(base_dir: str = "results"):
    """
    Find all sweep model directories.

    For each sweep directory, prefer final_model if it exists,
    otherwise use the checkpoint with the largest number.

    Args:
        base_dir: Base directory containing sweep results

    Returns:
        List of paths to model directories
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"⚠️  Base directory {base_dir} does not exist")
        return []

    # Find all sweep directories (sweep_* or *_sweep_*)
    sweep_dirs = sorted(base_path.glob("*sweep*"))

    # Check for final_model subdirectory or latest checkpoint
    model_paths = []
    for sweep_dir in sweep_dirs:
        final_model_path = sweep_dir / "final_model"

        if final_model_path.exists() and final_model_path.is_dir():
            # Prefer final_model if it exists
            model_paths.append(final_model_path)
            print(f"✅ Using final_model in {sweep_dir.name}")
        else:
            # Look for checkpoint-* directories
            checkpoint_dirs = list(sweep_dir.glob("checkpoint-*"))

            if checkpoint_dirs:
                # Extract checkpoint numbers and find the largest
                checkpoint_numbers = []
                for ckpt_dir in checkpoint_dirs:
                    match = re.search(r'checkpoint-(\d+)', ckpt_dir.name)
                    if match:
                        checkpoint_numbers.append((int(match.group(1)), ckpt_dir))

                if checkpoint_numbers:
                    # Sort by checkpoint number and get the largest
                    checkpoint_numbers.sort(key=lambda x: x[0], reverse=True)
                    largest_checkpoint = checkpoint_numbers[0][1]
                    model_paths.append(largest_checkpoint)
                    print(f"✅ Using {largest_checkpoint.name} in {sweep_dir.name}")
                else:
                    print(f"⚠️  No valid checkpoints found in {sweep_dir}")
            else:
                print(f"⚠️  No final_model or checkpoints found in {sweep_dir}")

    return model_paths


def load_model_and_tokenizer(model_path: Path, device: str = "auto"):
    """
    Load a model and tokenizer from a checkpoint.

    Args:
        model_path: Path to model checkpoint directory
        device: Device to load the model on ("auto", "cuda", "cpu")

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from: {model_path}")

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        "trust_remote_code": True,
    }

    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    model.eval()  # Set to evaluation mode

    if device == "cpu":
        model = model.to(device)

    print(f"✅ Model loaded successfully on {device}")
    return model, tokenizer


def benchmark_single_model(
    model_path: Path,
    preprompt_file: str,
    exam_path: str,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    openai_api_key: str = None,
    device: str = "auto"
):
    """
    Benchmark a single model and save results in its directory.

    Args:
        model_path: Path to model directory (e.g., results/sweep_lr1e-06_bs2_ep1/final_model)
        preprompt_file: Path to preprompt text file
        exam_path: Path to exam JSON file
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        openai_api_key: OpenAI API key for intelligent answer matching
        device: Device to run inference on

    Returns:
        dict: Benchmark results
    """
    # Load preprompt
    with open(preprompt_file, 'r', encoding='utf-8') as f:
        preprompt = f.read().strip()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    # Run benchmark
    print(f"\nRunning benchmark on {model_path.parent.name}...")
    results = run_single_benchmark(
        model=model,
        tokenizer=tokenizer,
        exam_path=exam_path,
        preprompt=preprompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        openai_api_key=openai_api_key
    )

    # Save results in the model's parent directory (sweep_* folder)
    output_dir = model_path.parent
    exam_name = Path(exam_path).stem
    output_file = output_dir / f"benchmark_results_{exam_name}.json"

    # Add metadata
    results["model_path"] = str(model_path)
    results["benchmark_timestamp"] = datetime.now().isoformat()
    results["preprompt_file"] = preprompt_file
    results["exam_path"] = exam_path

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Benchmark results saved to {output_file}")

    # Clean up model from memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all models from hyperparameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="results",
        help="Base directory containing sweep_* folders"
    )
    parser.add_argument(
        "--preprompt_file",
        type=str,
        required=True,
        help="Path to preprompt text file"
    )
    parser.add_argument(
        "--exam_path",
        type=str,
        required=True,
        help="Path to exam JSON file"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy decoding)"
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key for intelligent answer matching"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on"
    )

    # WandB arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=True,
        help="Enable WandB logging for benchmark comparison"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="Train-sft-sweep-benchmarks",
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="kejdi-lleshi-university-of-lausanne",
        help="WandB entity/username"
    )

    args = parser.parse_args()

    # Get OpenAI API key from argument or environment variable
    openai_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")

    # Find all sweep models
    print("="*80)
    print("BENCHMARKING SWEEP MODELS")
    print("="*80)
    print(f"Base directory: {args.base_dir}")
    print(f"Preprompt: {args.preprompt_file}")
    print(f"Exam: {args.exam_path}")
    print("="*80 + "\n")

    model_paths = find_sweep_models(args.base_dir)

    if not model_paths:
        print("❌ No sweep models found!")
        return

    print(f"Found {len(model_paths)} models to benchmark:\n")
    for i, path in enumerate(model_paths, 1):
        print(f"  {i}. {path.parent.name}")
    print()

    # Initialize WandB if enabled
    if args.use_wandb:
        exam_name = Path(args.exam_path).stem
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"benchmark_sweep_{exam_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "exam_path": args.exam_path,
                "preprompt_file": args.preprompt_file,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "num_models": len(model_paths)
            }
        )
        print(f"✅ WandB initialized: {args.wandb_project}")

    # Benchmark each model
    all_results = []
    successful = 0
    failed = 0

    for i, model_path in enumerate(model_paths, 1):
        print(f"\n{'='*80}")
        print(f"MODEL {i}/{len(model_paths)}: {model_path.parent.name}")
        print(f"{'='*80}")

        try:
            result = benchmark_single_model(
                model_path=model_path,
                preprompt_file=args.preprompt_file,
                exam_path=args.exam_path,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                openai_api_key=openai_api_key,
                device=args.device
            )

            model_name = model_path.parent.name
            result_summary = {
                "model": model_name,
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
                "exam_name": result["exam_name"],
                "success": True
            }
            all_results.append(result_summary)

            # Log to WandB
            if args.use_wandb:
                # Parse hyperparameters from model name (e.g., sweep_lr1e-06_bs2_ep1)
                match = re.search(r'lr([\de-]+)_bs(\d+)_ep(\d+)', model_name)
                if match:
                    lr_str, bs_str, ep_str = match.groups()
                    lr = float(lr_str)
                    bs = int(bs_str)
                    ep = int(ep_str)

                    wandb.log({
                        f"{model_name}/accuracy": result["accuracy"],
                        f"{model_name}/correct": result["correct"],
                        f"{model_name}/total": result["total"],
                        f"{model_name}/learning_rate": lr,
                        f"{model_name}/batch_size": bs,
                        f"{model_name}/epochs": ep,
                    })

                    # Also log with hyperparams as dimensions for easier comparison
                    wandb.log({
                        "model_name": model_name,
                        "learning_rate": lr,
                        "batch_size": bs,
                        "epochs": ep,
                        "accuracy": result["accuracy"],
                        "correct": result["correct"],
                        "total": result["total"],
                    })

            successful += 1
        except Exception as e:
            print(f"❌ Failed to benchmark {model_path.parent.name}: {e}")
            all_results.append({
                "model": model_path.parent.name,
                "error": str(e),
                "success": False
            })
            failed += 1

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Total models: {len(model_paths)}")
    print(f"Successfully benchmarked: {successful}")
    print(f"Failed: {failed}")
    print("\nResults:")

    # Sort by accuracy (successful ones first)
    successful_results = [r for r in all_results if r["success"]]
    failed_results = [r for r in all_results if not r["success"]]
    successful_results.sort(key=lambda x: x["accuracy"], reverse=True)

    for r in successful_results:
        print(f"  ✅ {r['model']}: {r['accuracy']:.2%} ({r['correct']}/{r['total']})")

    for r in failed_results:
        print(f"  ❌ {r['model']}: {r['error']}")

    print("="*80 + "\n")

    # Save summary to base directory
    summary_file = Path(args.base_dir) / "benchmark_summary.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "total_models": len(model_paths),
        "successful": successful,
        "failed": failed,
        "results": all_results,
        "exam_path": args.exam_path,
        "preprompt_file": args.preprompt_file
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"📊 Summary saved to {summary_file}")

    # Create WandB summary table
    if args.use_wandb and successful_results:
        # Create DataFrame for better visualization
        table_data = []
        for r in successful_results:
            # Parse hyperparameters from model name
            match = re.search(r'lr([\de-]+)_bs(\d+)_ep(\d+)', r["model"])
            if match:
                lr_str, bs_str, ep_str = match.groups()
                table_data.append({
                    "Model": r["model"],
                    "Learning Rate": float(lr_str),
                    "Batch Size": int(bs_str),
                    "Epochs": int(ep_str),
                    "Accuracy": r["accuracy"],
                    "Correct": r["correct"],
                    "Total": r["total"]
                })

        if table_data:
            df = pd.DataFrame(table_data)
            wandb.log({"benchmark_summary_table": wandb.Table(dataframe=df)})

            # Log summary metrics
            wandb.summary["best_accuracy"] = max(r["accuracy"] for r in successful_results)
            wandb.summary["worst_accuracy"] = min(r["accuracy"] for r in successful_results)
            wandb.summary["mean_accuracy"] = sum(r["accuracy"] for r in successful_results) / len(successful_results)
            wandb.summary["total_models_benchmarked"] = successful

            print(f"✅ Benchmark summary table logged to WandB")

        wandb.finish()
        print(f"✅ WandB run completed")


if __name__ == "__main__":
    main()
