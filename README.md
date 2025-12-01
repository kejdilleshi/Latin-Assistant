# Latin Language Model Fine-Tuning

A comprehensive framework for supervised fine-tuning (SFT) of large language models on Latin language tasks using DeepSpeed for distributed training.

## Project Overview

This project focuses on improving LLM performance on Latin language understanding through targeted fine-tuning on grammar-focused datasets. The key areas of improvement include:

1. **Translation** - Latin to modern languages and vice versa
2. **Syntactic Understanding** - Sentence structure and dependencies
3. **Grammar Mastery** - Case endings, verb moods, conjugations, and declensions
4. **Complex Constructions** - Participial clauses, gerunds, and coordination
5. **Natural Word Order** - Adapting Latin structure to target languages

## Project Structure

```
DeepSpeed/
│
├── sft_training/              # Main Python package
│   ├── models/                # Model setup and configuration
│   │   ├── __init__.py
│   │   └── model_utils.py     # Model initialization, layer freezing, dropout
│   ├── data/                  # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── data_utils.py      # Dataset loading, JSONL reading, preprocessing
│   ├── training/              # Training configuration
│   │   ├── __init__.py
│   │   └── trainer_utils.py   # SFTTrainer setup and configuration
│   └── tools/                 # Data preparation utilities
│       ├── __init__.py
│       └── build_sft_data.py  # GPT-4 powered data generation
│
├── scripts/                   # Executable training scripts
│   ├── train_sft.py          # Main training entry point
│   ├── run_hyperparameter_sweep.py  # Hyperparameter search
│   └── build_sft_data.py     # Data building CLI
│
├── benchmark/                 # Model evaluation
│   ├── run_benchmark.py      # Single model benchmarking
│   ├── benchmark_sweep_models.py  # Batch evaluation
│   └── job.sbatch            # SLURM script for benchmarking
│
├── Configuration Files
│   ├── setup.py              # Package installation
│   ├── pyproject.toml        # Modern Python packaging
│   ├── deepspeed_config.json # DeepSpeed ZeRO configuration
│   ├── requirements.txt      # Python dependencies
│   ├── hostfile              # Multi-node host configuration
│   └── *.jinja               # Chat templates for tokenization
│
└── SLURM Scripts
    ├── sweep.sbatch          # Hyperparameter sweep job
    └── deepspeed.sbatch      # Distributed training job
```

## Features

### Training Capabilities
- **Distributed Training** with DeepSpeed ZeRO-3 optimization
- **Sequence Packing** for efficient training on variable-length sequences
- **Gradient Checkpointing** to reduce memory footprint
- **Flash Attention 2** for faster attention computation
- **Selective Layer Freezing** to fine-tune only top layers
- **Weights & Biases** integration for experiment tracking

### Data Processing
- **Automated Data Generation** using GPT-4 API
- **Multi-task Learning** with translation, morphosyntax, and transformations
- **JSONL Format** for efficient data loading
- **Train/Val/Test Splits** with configurable ratios

### Model Support
- Llama 3.1 (8B and larger)
- SmolLM family
- Mistral models
- Any HuggingFace CausalLM model

## Quick Start

### Installation

```bash
# Clone repository
cd /path/to/DeepSpeed

# Install package in development mode
pip install -e .

# Or install with dependencies
pip install -r requirements.txt
```

### Training a Model

**Single GPU/Node:**
```bash
python scripts/train_sft.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name local \
    --local_data_path data/sft_items.jsonl \
    --output_dir ./outputs/llama-latin \
    --per_device_train_batch_size 2 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --packing \
    --use_wandb
```

**Multi-GPU with DeepSpeed (SLURM):**
```bash
sbatch deepspeed.sbatch
```

### Hyperparameter Sweep

Run automated hyperparameter search across learning rates, batch sizes, and epochs:

```bash
python scripts/run_hyperparameter_sweep.py

# Or submit to SLURM
sbatch sweep.sbatch
```

### Building Training Data

Generate training data from Latin text using GPT-4:

```bash
python scripts/build_sft_data.py \
    data/latin_corpus.txt \
    --outdir data/processed \
    --max_items 100 \
    --model gpt-4o-2024-08-06
```

## Configuration

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model identifier |
| `--dataset_name` | `local` | Dataset name or "local" for JSONL |
| `--local_data_path` | `data/sft_items.jsonl` | Path to local training data |
| `--output_dir` | Required | Output directory for checkpoints |
| `--per_device_train_batch_size` | `1` | Batch size per GPU |
| `--num_train_epochs` | `1.0` | Number of training epochs |
| `--learning_rate` | `1e-5` | Learning rate |
| `--packing` | `True` | Enable sequence packing |
| `--use_wandb` | `True` | Enable W&B logging |
| `--deepspeed` | `./deepspeed_config.json` | DeepSpeed config path |

### DeepSpeed Configuration

The project uses ZeRO-3 optimization with:
- **Stage 3** - Optimizer state, gradient, and parameter partitioning
- **Offload** - CPU offloading for large models
- **BF16** - Mixed precision training
- **Gradient Accumulation** - Effective larger batch sizes

Edit `deepspeed_config.json` to customize.

## Data Format

Training data should be in JSONL format with messages:

```json
{
  "messages": [
    {"role": "user", "content": "Translate: Caesar Galliam vicit"},
    {"role": "assistant", "content": "Caesar conquered Gaul"}
  ]
}
```

Or prompt/target format (auto-converted):

```json
{
  "prompt": "Translate: Caesar Galliam vicit",
  "target": "Caesar conquered Gaul"
}
```

## Training Tasks

The data generation pipeline creates diverse training tasks:

1. **Translation** - Idiomatic Latin ↔ English translation
2. **Morphosyntax Analysis** - Token-level POS, lemma, and dependency parsing
3. **Syntactic Transformations** - Active/passive voice, tense changes, etc.
4. **Clause Identification** - Main clause extraction
5. **Complement Detection** - Object and complement identification

## Hyperparameter Sweep

Default sweep configuration:
- **Learning Rates**: 1e-6, 5e-6, 1e-5
- **Batch Sizes**: 2, 4
- **Epochs**: 1, 2
- **Packing**: Enabled

Customize in `scripts/run_hyperparameter_sweep.py`.

## SLURM Configuration

### Resource Requirements

| Script | GPUs | Memory | Time | Purpose |
|--------|------|--------|------|---------|
| `sweep.sbatch` | 6 | 160GB | 24h | Hyperparameter search |
| `deepspeed.sbatch` | 4 | 160GB | 1h | Single training run |

### Environment Variables

```bash
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_key"
export OPENAI_API_KEY="your_openai_key"  # For data generation
```

## Benchmarking

Evaluate trained models on test sets:

```bash
python benchmark/run_benchmark.py \
    --model_path ./outputs/llama-latin/final_model \
    --exam_path data/test_set.json \
    --preprompt_file data/preprompt.txt \
    --max_new_tokens 100 \
    --temperature 0.0
```

Batch evaluation across multiple checkpoints:

```bash
python benchmark/benchmark_sweep_models.py \
    --base_dir /scratch/experiments \
    --exam_path data/test_set.json
```

## Python API

Use as a package in your own code:

```python
from sft_training import setup_model, setup_tokenizer
from sft_training.data import load_and_split_datasets
from sft_training.training import create_trainer

# Setup
tokenizer = setup_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
model = setup_model(args)

# Load data
train_ds, test_ds, val_ds = load_and_split_datasets(args, tokenizer)

# Train
trainer = create_trainer(model, tokenizer, train_ds, val_ds, args)
trainer.train()
```

## Model Outputs

Trained models are saved with:
- Model weights (DeepSpeed format or consolidated)
- Tokenizer configuration
- Training metrics and logs
- W&B experiment tracking

## Best Practices

### Training
- Start with small learning rates (1e-6 to 1e-5)
- Use gradient accumulation for larger effective batch sizes
- Enable packing for variable-length sequences
- Monitor validation loss to prevent overfitting
- Use assistant-only loss masking for chat models

### Data
- Ensure balanced representation across tasks
- Include diverse sentence structures
- Validate data quality before training
- Use train/val/test splits (80/10/10)

### Infrastructure
- Use DeepSpeed for models > 7B parameters
- Enable gradient checkpointing to save memory
- Use Flash Attention 2 for faster training
- Monitor GPU memory and adjust batch size accordingly

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size`
- Enable gradient checkpointing
- Use DeepSpeed ZeRO-3 with CPU offload
- Disable packing

### Slow Training
- Increase `per_device_train_batch_size`
- Enable gradient accumulation
- Use larger GPUs (A100, L40)
- Ensure Flash Attention 2 is installed

### Poor Results
- Increase training epochs
- Try different learning rates
- Check data quality and balance
- Use larger models
- Adjust learning rate schedule

## Citation

If you use this code in your research, please cite:

```bibtex
@software{latin_sft_2025,
  title = {Latin Language Model Fine-Tuning Framework},
  author = {Lleshi, Kejdi},
  year = {2025},
  institution = {University of Lausanne}
}
```

## License

[Specify your license here]

## Contact

For questions and support:
- Email: Kejdi.Lleshi@unil.ch
- GitHub Issues: [Your repository URL]

## Acknowledgments

- DeepSpeed team for distributed training framework
- Hugging Face for transformers and TRL libraries
- OpenAI for GPT-4 API used in data generation
- University of Lausanne for computational resources
