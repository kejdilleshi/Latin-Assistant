#!/usr/bin/env python
"""
Standalone benchmark script for evaluating trained models.

This script loads a trained model checkpoint and evaluates it on one or more exam datasets.
It supports multiple preprompts and exam files, and saves detailed results.

Usage examples:
    # Benchmark with a HuggingFace base model
    python run_benchmark.py --model_path HuggingFaceTB/SmolLM3-3B \
                            --exam_path /path/to/exam.json \
                            --preprompt_file /path/to/preprompt.txt

    # Benchmark with a trained checkpoint
    python run_benchmark.py --model_path results/SmolLM3_Latin_sft_packing/final_model \
                            --exam_path /path/to/exam.json \
                            --preprompt_file /path/to/preprompt.txt

    # With custom generation parameters
    python run_benchmark.py --model_path HuggingFaceTB/SmolLM3-3B \
                            --exam_path /path/to/exam.json \
                            --preprompt_file /path/to/preprompt.txt \
                            --max_new_tokens 50 \
                            --temperature 0.0
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmark import benchmark_exam


def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """
    Load a model and tokenizer from a checkpoint or HuggingFace model name.

    Args:
        model_path: Path to model checkpoint directory OR HuggingFace model name
                   (e.g., "HuggingFaceTB/SmolLM3-3B" or "results/my_model/final_model")
        device: Device to load the model on ("auto", "cuda", "cpu")

    Returns:
        tuple: (model, tokenizer)
    """
    # Check if it's a local path or HuggingFace model name
    is_local = os.path.exists(model_path) and os.path.isdir(model_path)

    if is_local:
        print(f"Loading model from local checkpoint: {model_path}")
    else:
        print(f"Loading model from HuggingFace Hub: {model_path}")

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

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()  # Set to evaluation mode

    if device == "cpu":
        model = model.to(device)

    print(f"✅ Model loaded successfully on {device}")
    print(f"   Model type: {model.config.model_type}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model, tokenizer


def load_preprompt(preprompt_path: str) -> str:
    """Load preprompt text from file."""
    with open(preprompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def run_single_benchmark(
    model,
    tokenizer,
    exam_path: str,
    preprompt: str,
    max_new_tokens: int = 20,
    temperature: float = 0.2,
    openai_api_key: str = None
):
    """
    Run benchmark on a single exam file.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        exam_path: Path to the exam JSON file
        preprompt: Preprompt text to use
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        openai_api_key: OpenAI API key for intelligent answer matching (optional)

    Returns:
        dict: Benchmark results including accuracy and detailed predictions
    """
    exam_name = Path(exam_path).stem
    print(f"\n{'='*70}")
    print(f"Running benchmark: {exam_name}")
    if openai_api_key:
        print(f"Using OpenAI API for intelligent answer matching")
    else:
        print(f"Using simple first-character matching")
    print(f"{'='*70}")

    accuracy, detailed_results = benchmark_exam(
        model=model,
        tokenizer=tokenizer,
        data_path=exam_path,
        preprompt=preprompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        openai_api_key=openai_api_key
    )

    return {
        "exam_name": exam_name,
        "exam_path": exam_path,
        "accuracy": accuracy,
        "correct": sum(1 for r in detailed_results if r["is_correct"]),
        "total": len(detailed_results),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "detailed_results": detailed_results
    }



def main():
    parser = argparse.ArgumentParser(
        description="Benchmark trained models on exam datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint directory OR HuggingFace model name (e.g., 'HuggingFaceTB/SmolLM3-3B')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on"
    )

    # Benchmark configuration (single exam mode)
    parser.add_argument(
        "--exam_path",
        type=str,
        help="Path to exam JSON file (for single exam mode)"
    )
    parser.add_argument(
        "--preprompt_file",
        type=str,
        help="Path to preprompt text file (for single exam mode)"
    )

    # Benchmark configuration (multi-exam mode)
    parser.add_argument(
        "--benchmark_config",
        type=str,
        help="Path to JSON config file with multiple benchmark configurations"
    )

    # Generation parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0.0 for greedy decoding)"
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key for intelligent answer matching (optional, uses env OPENAI_API_KEY if not provided)"
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.benchmark_config and not (args.exam_path and args.preprompt_file):
        parser.error("Either --benchmark_config or both --exam_path and --preprompt_file must be provided")

    # Get OpenAI API key from argument or environment variable
    openai_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        print(f"✅ OpenAI API key found - will use GPT for intelligent answer matching")
    else:
        print(f"ℹ️  No OpenAI API key provided - will use simple first-character matching")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    # Prepare benchmark configurations
    if args.benchmark_config:
        # Load from config file
        with open(args.benchmark_config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        benchmark_configs = config_data.get("benchmarks", [])
        print(f"Loaded {len(benchmark_configs)} benchmark configurations from {args.benchmark_config}")
    else:
        # Single benchmark mode
        benchmark_configs = [{
            "exam_path": args.exam_path,
            "preprompt_file": args.preprompt_file,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature
        }]

    # Load preprompt
    preprompt = load_preprompt(args.preprompt_file)

    # Run benchmarks
    results = run_single_benchmark(
        model,
        tokenizer,
        exam_path=args.exam_path,
        preprompt=preprompt,
        max_new_tokens=args.max_new_tokens,
        temperature=0,
        openai_api_key=openai_api_key
    )
    print("\nBenchmarking completed. Results:", results)


if __name__ == "__main__":
    main()
