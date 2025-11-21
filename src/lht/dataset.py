"""
Data pipeline for LHT.

This module handles:
- loading raw text from HuggingFace datasets (or your custom ones),
- tokenization,
- packing into fixed-length sequences,
- providing attention masks and any auxiliary labels.

For LHT, we *don't* need gold sentence/section boundaries for training,
but you may still want them for evaluation or analysis.

Matches HDT's pretraining setup:
- Multiple corpora: unarXive, HUPD, Wikipedia
- Interleaved sampling with configurable probabilities
- 8192 token sequences for long-document modeling
"""

from typing import Any, Dict, Tuple

from datasets import interleave_datasets, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


def build_tokenizer(name_or_path: str) -> PreTrainedTokenizer:
    """Load a HuggingFace tokenizer."""
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    return tok


def load_lht_pretrain_dataset(cfg) -> Tuple[Any, PreTrainedTokenizer]:
    """
    Load a dataset compatible with LHT pretraining, returning a
    HuggingFace Dataset or IterableDataset with tokenized text.

    Supports both single dataset (backward compat) and multiple datasets
    with interleaved sampling (HDT-style).
    """
    tokenizer = build_tokenizer(cfg.data.tokenizer_name_or_path)

    # Check if using multi-source HDT-style setup
    if cfg.data.sources is not None and len(cfg.data.sources) > 0:
        # Load multiple datasets and interleave
        datasets = []
        for source_name in cfg.data.sources:
            print(f"Loading dataset: {source_name}")
            ds = load_dataset(source_name, split="train", streaming=True)
            datasets.append(ds)

        # Interleave with specified sampling probabilities
        probs = cfg.data.sampling_probs
        if probs is None:
            probs = [1.0 / len(datasets)] * len(datasets)

        print(f"Interleaving {len(datasets)} datasets with probs: {probs}")
        raw = interleave_datasets(
            datasets,
            probabilities=probs,
            seed=cfg.seed,
            stopping_strategy="all_exhausted",
        )
    else:
        # Single dataset (backward compatibility)
        dataset_name = cfg.data.dataset_name or "howey/wiki_en"
        print(f"Loading single dataset: {dataset_name}")
        raw = load_dataset(dataset_name, split="train", streaming=True)

    def tokenize_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
        texts = examples[cfg.data.text_column]
        out = tokenizer(
            texts,
            truncation=True,
            max_length=cfg.data.max_seq_len,
            padding="max_length",
            return_attention_mask=True,
        )
        return out

    # For streaming datasets, map on the fly
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=[cfg.data.text_column],
    )

    return tokenized, tokenizer
