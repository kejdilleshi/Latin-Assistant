"""Model setup and configuration utilities for SFT training."""

import math
import operator
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    """Set dropout probability across all dropout layers in the model."""
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

    # Print a small report
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[Freeze Report] Found layers at '{path}' with {n_total} blocks. "
        f"Froze bottom {n_freeze} ({fraction*100:.1f}%). "
        f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M."
    )


def setup_model(args):
    """Initialize and configure the model with gradient checkpointing and layer freezing."""
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else None,
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2",  # Required for packing
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # Important for training + gradient checkpointing
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.config.gradient_checkpointing = True

    # Freeze bottom 50% of transformer layers
    # freeze_bottom_fraction_of_layers(model, fraction=0.50)

    return model


def setup_tokenizer(model_name: str):
    """Initialize tokenizer and set padding token if needed."""
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    
    # Load the patched chat template
    with open("chat_template_llama31_masked.jinja", "r") as f:
        tokenizer.chat_template = f.read()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    return tokenizer
