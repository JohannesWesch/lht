"""
Configuration loading utilities for LHT.

This module defines a simple dataclass-based config and a loader
from YAML files, so you can keep experiments reproducible.
"""

from dataclasses import dataclass
from typing import List, Optional

import yaml


@dataclass
class MLSWAConfig:
    """Configuration for Multi-Level Sliding Window Attention (ML-SWA) hierarchy.

    Controls which hierarchy levels are active in each layer and their window sizes:
    - Early layers: tokens only (max_level=0), window of 256 tokens
    - Middle layers: tokens + sentences (max_level=1), 256 tokens + 64 sentences
    - Deep layers: all levels, 256 tokens + 64 sentences + 16 sections

    Window sizes control enumeration distance within each level.
    All positions attend via level 0; boundary positions add higher-level windows (OR merge).
    """

    num_levels: int  # e.g. 3 → 0=token, 1=sent, 2=sec
    window_size_per_level: List[int]  # [256, 64, 16] = tokens, sentences, sections
    layer_max_level: List[int]  # len = num_layers, max active level per layer


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    num_layers: int  # total number of transformer layers
    d_ff: int
    dropout: float
    max_seq_len: int
    rope: bool = True
    mlswa: MLSWAConfig = None  # Multi-Level Sliding Window Attention config


@dataclass
class TrainingConfig:
    task: str
    batch_size: int
    grad_accum_steps: int
    num_steps: int
    warmup_steps: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    log_every: int
    save_every: int
    eval_every: int
    mixed_precision: str
    mlm_probability: float = 0.15


@dataclass
class DataConfig:
    text_column: str
    num_workers: int
    shuffle_buffer_size: int
    max_seq_len: int
    tokenizer_name_or_path: str
    pretokenized: bool = False
    sources: Optional[List[str]] = None
    sampling_probs: Optional[List[float]] = None
    dataset_name: Optional[str] = None  # for backward compatibility
    ds_info: Optional[List[dict]] = None
    model_max_length: Optional[int] = None
    tok_name: Optional[str] = None


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""

    project: str
    entity: Optional[str] = None
    name: Optional[str] = None  # defaults to experiment_name
    tags: Optional[List[str]] = None
    log_model: bool = False  # save checkpoints to wandb artifacts
    watch_model: str = "gradients"  # "gradients", "all", or "none"
    watch_freq: int = 1000  # log gradients/params every N steps
    offline: bool = False  # run in offline mode


@dataclass
class ExperimentConfig:
    experiment_name: str
    seed: int
    device: str
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    wandb: WandbConfig


def load_config(path: str) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Parse wandb config
    wandb_raw = raw.get("wandb", {})
    wandb_cfg = WandbConfig(**wandb_raw) if wandb_raw else WandbConfig(project="lht")

    # Parse model config with nested mlswa
    model_raw = raw["model"]
    mlswa_raw = model_raw.get("mlswa")
    if mlswa_raw:
        mlswa_cfg = MLSWAConfig(**mlswa_raw)
        # Remove mlswa from model_raw so it doesn't conflict
        model_raw_copy = {k: v for k, v in model_raw.items() if k != "mlswa"}
        model_cfg = ModelConfig(**model_raw_copy, mlswa=mlswa_cfg)
    else:
        model_cfg = ModelConfig(**model_raw)

    # Ensure numeric training values are properly typed
    training_raw = raw["training"].copy()
    training_raw["learning_rate"] = float(training_raw["learning_rate"])
    training_raw["weight_decay"] = float(training_raw["weight_decay"])
    training_raw["mlm_probability"] = float(training_raw.get("mlm_probability", 0.15))

    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        seed=raw["seed"],
        device=raw["device"],
        model=model_cfg,
        training=TrainingConfig(**training_raw),
        data=DataConfig(**raw["data"]),
        wandb=wandb_cfg,
    )
