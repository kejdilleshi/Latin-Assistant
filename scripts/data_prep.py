# data_prep.py
import os
from typing import Optional, Iterable, List, Dict, Any
from typing import List, Tuple
from transformers import PreTrainedTokenizerBase
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


def _group_token_list_into_blocks(
    input_ids: List[int],
    max_seq_length: int,
) -> List[List[int]]:
    """Slice a long list of token ids into contiguous fixed-length blocks."""
    usable_len = (len(input_ids) // max_seq_length) * max_seq_length
    if usable_len == 0:
        return []
    return [
        input_ids[i : i + max_seq_length]
        for i in range(0, usable_len, max_seq_length)
    ]


def _tokens_to_dataset(
    blocks: List[List[int]],
    max_seq_length: int,
    ensure_attention_mask: bool = True,
) -> Dataset:
    """Turn fixed-length blocks into a HF Dataset with input_ids (+ attention_mask)."""
    if not blocks:
        raise ValueError(
            "No token blocks were produced. "
            "Check that your corpus has enough tokens for the chosen max_seq_length."
        )
    result = {"input_ids": blocks}
    if ensure_attention_mask:
        result["attention_mask"] = [[1] * max_seq_length for _ in blocks]
    return Dataset.from_dict(result)


def _tokenize_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    add_special_tokens: bool = False,
) -> List[int]:
    """Tokenize a raw string and return token ids (no attention mask)."""
    tokens = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
    )
    return tokens["input_ids"]


def _read_local_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local corpus not found at: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()
    # optional light normalization
    return " ".join(raw_text.split())


def _iter_stream_texts(iterable_ds: Iterable[Dict[str, Any]], take: Optional[int]) -> List[str]:
    """Collect a finite number of text samples from a streaming dataset."""
    samples: List[str] = []
    if take is None:
        # default safety: cap if not provided
        take = 200
    for i, ex in enumerate(iterable_ds):
        if "text" in ex and isinstance(ex["text"], str):
            samples.append(ex["text"])
        if i + 1 >= take:
            break
    return samples


def _collect_nonstream_texts(hf_ds) -> List[str]:
    """Collect all texts from a non-streaming HF dataset split."""
    # Expect column 'text'. If different, modify as needed.
    if "text" not in hf_ds.column_names:
        raise ValueError(f"Expected a 'text' column in the dataset, found: {hf_ds.column_names}")
    return hf_ds["text"]


def prepare_training_dataset(
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    # If provided, we use the local .txt file. If None, we fall back to a HF dataset.
    path: Optional[str] = None,
    # HF dataset fallback options (used only if path is None):
    dataset_name: str = "HuggingFaceFW/fineweb-2",
    dataset_config: Optional[str] = "lat_Latn",
    split: str = "train",
    streaming: bool = True,
    take: Optional[int] = 200,
) -> Dataset:
    """
    Prepare a HF Dataset of fixed-length token blocks for causal LM training.

    Args
    ----
    tokenizer: PreTrainedTokenizerBase
        Already loaded tokenizer (pad_token should be set externally if needed).
    max_seq_length: int
        Block size for training.
    path: Optional[str]
        Local .txt path. If provided, we train from this corpus. If None, we load from HF.
    dataset_name: str
        HF dataset repository id (default: FineWeb-2).
    dataset_config: Optional[str]
        HF dataset config (e.g., language filter for FineWeb-2).
    split: str
        HF split to load.
    streaming: bool
        If True, use streaming for HF datasets (and take `take` samples). If False, load fully.
    take: Optional[int]
        Number of samples to gather from a streaming dataset. Ignored if streaming=False.

    Returns
    -------
    Dataset
        A Dataset with fields: input_ids (List[List[int]]), attention_mask (List[List[int]]).
    """
    # 1) Build a single long text string
    if path is not None:
        # Local .txt mode
        raw_text = _read_local_text(path)
    else:
        # Hugging Face dataset mode
        if streaming:
            hf_iterable = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
            texts = _iter_stream_texts(hf_iterable, take=take)
        else:
            hf_ds = load_dataset(dataset_name, dataset_config, split=split, streaming=False)
            texts = _collect_nonstream_texts(hf_ds)
        if len(texts) == 0:
            raise ValueError("No text samples were loaded from the Hugging Face dataset.")
        raw_text = " ".join(" ".join(t.split()) for t in texts)

    # 2) Tokenize once (no special tokens per chunk)
    input_ids_all = _tokenize_text(tokenizer, raw_text, add_special_tokens=False)

    # 3) Slice into fixed-length blocks
    blocks = _group_token_list_into_blocks(input_ids_all, max_seq_length)

    # 4) Build Dataset with attention masks
    ds = _tokens_to_dataset(blocks, max_seq_length, ensure_attention_mask=True)
    return ds
# --- Preview utilities (append to data_prep.py) ---



def detok_sample(
    tokenizer: PreTrainedTokenizerBase,
    input_ids: List[int],
    max_chars: int = 400,
) -> str:
    """
    Decode a single block of token ids to text for human inspection.
    Truncates display to max_chars for cleanliness.
    """
    text = tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " …"
    return text

def preview_training_samples(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    n: int = 3,
    max_chars: int = 400,
    show_ids: bool = False,
) -> List[Tuple[str, List[int]]]:
    """
    Return a small list of (decoded_text, input_ids) for the first n blocks.
    Set show_ids=True if you also want to log/print the raw ids.
    """
    n = min(n, len(ds))
    previews = []
    for i in range(n):
        ids = ds[i]["input_ids"]
        txt = detok_sample(tokenizer, ids, max_chars=max_chars)
        previews.append((txt, ids if show_ids else []))
    return previews