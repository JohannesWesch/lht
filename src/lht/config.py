"""
Configuration loading utilities for LHT.

This module defines a simple dataclass-based config and a loader
from YAML files, so you can keep experiments reproducible.
"""

from dataclasses import dataclass
from typing import List, Optional

import yaml


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


@dataclass
class HierarchyLevelConfig:
    """Config for a single level of the learned hierarchy."""

    name: str  # e.g., "sentence", "paragraph", "section"
    at_layer: int  # which layer to run this router after (1-indexed)
    target_head_ratio: float  # target proportion of head tokens
    loss_weight: float  # weight for ratio loss


@dataclass
class RouterConfig:
    """Config for the router network architecture."""

    hidden_dim: int
    window_size: int
    use_gumbel_ste: bool


@dataclass
class HierarchyConfig:
    """Config for the full learned hierarchy (K levels)."""

    levels: List[HierarchyLevelConfig]  # K abstraction levels
    router: RouterConfig  # shared router architecture


@dataclass
class AttentionConfig:
    local_window_tokens: int
    neighbour_sentences: int
    neighbour_sections: int
    use_doc_token: bool


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
    hierarchy: HierarchyConfig
    attention: AttentionConfig
    training: TrainingConfig
    data: DataConfig
    wandb: WandbConfig


def load_config(path: str) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Parse hierarchy levels
    hierarchy_raw = raw["hierarchy"]
    levels = [HierarchyLevelConfig(**lvl) for lvl in hierarchy_raw["levels"]]
    router_cfg = RouterConfig(**hierarchy_raw["router"])
    hierarchy = HierarchyConfig(levels=levels, router=router_cfg)

    # Parse wandb config
    wandb_raw = raw.get("wandb", {})
    wandb_cfg = WandbConfig(**wandb_raw) if wandb_raw else WandbConfig(project="lht")

    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        seed=raw["seed"],
        device=raw["device"],
        model=ModelConfig(**raw["model"]),
        hierarchy=hierarchy,
        attention=AttentionConfig(**raw["attention"]),
        training=TrainingConfig(**raw["training"]),
        data=DataConfig(**raw["data"]),
        wandb=wandb_cfg,
    )
