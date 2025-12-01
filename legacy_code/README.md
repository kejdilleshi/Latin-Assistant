# Legacy Code

This directory contains old/deprecated code files that have been replaced by the new package structure.

## ⚠️ Do Not Use These Files

These files are kept for reference only and **should not be used** in new code. They have been superseded by the organized package structure in `sft_training/` and `scripts/`.

## Files in This Directory

### Deprecated Utility Modules (Replaced by sft_training/)

| Old File | New Location | Purpose |
|----------|--------------|---------|
| `model_utils.py` | `sft_training/models/model_utils.py` | Model setup and configuration |
| `data_utils.py` | `sft_training/data/data_utils.py` | Data loading and preprocessing |
| `trainer_utils.py` | `sft_training/training/trainer_utils.py` | Trainer configuration |
| `build_sft_data.py` | `sft_training/tools/build_sft_data.py` | Data preparation tool |

### Deprecated Scripts (Replaced by scripts/)

| Old File | New Location | Purpose |
|----------|--------------|---------|
| `train_sft.py` | `scripts/train_sft.py` | Main training script |
| `run_hyperparameter_sweep.py` | `scripts/run_hyperparameter_sweep.py` | Hyperparameter sweep |

### Old/Experimental Training Scripts

| File | Status | Notes |
|------|--------|-------|
| `train_sft_legacy.py` | Deprecated | Old version of training script |
| `train_sft_mistral.py` | Deprecated | Mistral-specific training (obsolete) |
| `train_smol3.py` | Deprecated | SmolLM3-specific training (obsolete) |

### Utility Scripts (Excess/Experimental)

| File | Status | Notes |
|------|--------|-------|
| `decode_ids.py` | Utility | Token ID decoder (excess) |
| `inspect_token_length.py` | Utility | Token length inspector (excess) |
| `plot_loss.py` | Utility | Loss plotting script (excess) |
| `test_load.py` | Test | Loading test script (excess) |

## Migration Guide

If you have scripts that use these old files, update them to use the new structure:

### Old Import Style
```python
from model_utils import setup_model, setup_tokenizer
from data_utils import load_and_split_datasets
from trainer_utils import create_trainer
```

### New Import Style
```python
from sft_training import setup_model, setup_tokenizer, load_and_split_datasets, create_trainer

# Or module-level imports
from sft_training.models import setup_model, setup_tokenizer
from sft_training.data import load_and_split_datasets
from sft_training.training import create_trainer
```

### Old Script Paths
```bash
python train_sft.py
python run_hyperparameter_sweep.py
python build_sft_data.py
```

### New Script Paths
```bash
python scripts/train_sft.py
python scripts/run_hyperparameter_sweep.py
python scripts/build_sft_data.py
```

## When Can These Files Be Deleted?

These files can be safely deleted after:

1. ✅ All workflows are using the new package structure
2. ✅ All SLURM batch scripts reference `scripts/` directory
3. ✅ No custom scripts import from these old files
4. ✅ Testing confirms everything works with the new structure

## Verification Before Deletion

Before deleting these files, run these checks:

```bash
# Check for any references in batch scripts
grep -r "model_utils\|data_utils\|trainer_utils" *.sbatch

# Check for any custom scripts importing old modules
grep -r "from model_utils\|from data_utils\|from trainer_utils" .

# Check for any scripts using old paths
grep -r "train_sft.py\|run_hyperparameter_sweep.py" *.sbatch | grep -v "scripts/"
```

If all checks return empty or only show `legacy_code/` references, it's safe to delete this directory.

## Recommended Action

**Keep this directory for 1-2 weeks** to ensure everything works smoothly with the new structure, then delete it.

## Related Documentation

- [README_PACKAGE.md](../README_PACKAGE.md) - New package documentation
- [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) - Migration instructions
- [UPDATE_COMPLETE.md](../UPDATE_COMPLETE.md) - Complete update summary

---

**Moved to legacy_code/:** December 1, 2025
**Can be deleted after:** December 15, 2025 (pending verification)
