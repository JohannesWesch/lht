"""
Data pipeline for LHT.

This module handles:
- loading raw text from HuggingFace datasets (or your custom ones),
- tokenization,
- packing into fixed-length sequences,
- providing attention masks and any auxiliary labels.

For LHT, we *don't* need gold sentence/section boundaries for training,
but you may still want them for evaluation or analysis.
"""

from typing import Any, Dict

from datasets import load_dataset
from transformers import AutoTokenizer


def build_tokenizer(name_or_path: str):
    """Load a HuggingFace tokenizer."""
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    return tok


def load_lht_pretrain_dataset(cfg) -> Any:
    """
    Load a dataset compatible with LHT pretraining, returning a
    HuggingFace Dataset or IterableDataset with tokenized text.
    """
    raw = load_dataset(cfg.data.dataset_name)

    tokenizer = build_tokenizer(cfg.data.tokenizer_name_or_path)

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

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        num_proc=cfg.data.num_workers,
        remove_columns=raw["train"].column_names,
    )

    return tokenized, tokenizer
