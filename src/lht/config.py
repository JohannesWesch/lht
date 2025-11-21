"""
Configuration loading utilities for LHT.

This module defines a simple dataclass-based config and a loader
from YAML files, so you can keep experiments reproducible.
"""

from dataclasses import dataclass

import yaml


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers_local: int
    n_layers_mid: int
    n_layers_global: int
    d_ff: int
    dropout: float
    max_seq_len: int
    rope: bool = True


@dataclass
class RouterLevelConfig:
    target_head_ratio: float
    loss_weight: float


@dataclass
class RouterConfig:
    hidden_dim: int
    window_size: int
    use_gumbel_ste: bool
    token_to_sentence: RouterLevelConfig
    sentence_to_section: RouterLevelConfig


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


@dataclass
class DataConfig:
    dataset_name: str
    text_column: str
    num_workers: int
    shuffle_buffer_size: int
    max_seq_len: int
    tokenizer_name_or_path: str
    pretokenized: bool = False


@dataclass
class ExperimentConfig:
    experiment_name: str
    seed: int
    device: str
    model: ModelConfig
    router: RouterConfig
    attention: AttentionConfig
    training: TrainingConfig
    data: DataConfig


def load_config(path: str) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Minimal, direct mapping; you can make this more robust later.
    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        seed=raw["seed"],
        device=raw["device"],
        model=ModelConfig(**raw["model"]),
        router=RouterConfig(
            hidden_dim=raw["router"]["hidden_dim"],
            window_size=raw["router"]["window_size"],
            use_gumbel_ste=raw["router"]["use_gumbel_ste"],
            token_to_sentence=RouterLevelConfig(**raw["router"]["token_to_sentence"]),
            sentence_to_section=RouterLevelConfig(
                **raw["router"]["sentence_to_section"]
            ),
        ),
        attention=AttentionConfig(**raw["attention"]),
        training=TrainingConfig(**raw["training"]),
        data=DataConfig(**raw["data"]),
    )
