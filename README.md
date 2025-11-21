# Learned Hierarchical Transformer (LHT)

A hierarchical transformer encoder implementation combining:
- **H-Net-style routing**: Learn sentence and section boundaries without supervision
- **HDT-style hierarchical attention**: Child/parent/sibling attention patterns via xFormers
- **Multi-scale processing**: Local → Mid → Global layers with different attention patterns

## Installation

```bash
uv sync
```

For development dependencies (black, isort, ruff, pytest, ipykernel):

```bash
uv sync --extra dev
```

## Usage

### Training

```bash
lht-train-pretrain --config configs/pretrain_base.yaml
```

Or via script:

```bash
python scripts/train_pretrain.py --config configs/pretrain_base.yaml
```

### Evaluation

```bash
lht-eval --config configs/pretrain_base.yaml
```

## Architecture

The LHT encoder has three stages:

1. **Local Layers (1-3)**: Sliding window attention over tokens
2. **Mid Layers (4-6)**: Sentence-aware attention after first router
3. **Global Layers (7-12)**: Full hierarchical attention with learned sentence & section structure

### Routers

- `TokenToSentenceRouter`: Learns to identify sentence boundaries using straight-through estimation
- `SentenceToSectionRouter`: Learns section boundaries from sentence heads

### Attention Biases

- `build_local_attention_bias()`: Sliding window for local layers
- `SentenceAwareBias`: Restricts attention to sentence scope + heads
- `HierarchicalBias`: Full HDT-style child/parent/neighbour attention

## Project Structure

```
lht/
├── pyproject.toml           # Dependencies and build config
├── README.md
├── configs/
│   └── pretrain_base.yaml   # Example training configuration
├── src/lht/
│   ├── __init__.py
│   ├── config.py            # Dataclass-based config loader
│   ├── model.py             # LHTEncoder main model
│   ├── routers.py           # Boundary prediction routers
│   ├── attention_masks.py   # xFormers attention biases
│   ├── dataset.py           # HuggingFace dataset pipeline
│   ├── training.py          # Training utilities (ratio loss, etc.)
│   ├── utils.py             # Seeding, checkpointing
│   └── main.py              # CLI entrypoints
└── scripts/
    ├── train_pretrain.py    # Training wrapper
    └── eval.py              # Evaluation wrapper
```

## Configuration

See `configs/pretrain_base.yaml` for a full example. Key sections:

- `model`: Architecture hyperparameters (d_model, n_heads, layer counts)
- `router`: Router config (hidden_dim, target ratios, loss weights)
- `attention`: Attention patterns (window sizes, neighbour counts)
- `training`: Optimization settings (batch_size, learning_rate, steps)
- `data`: Dataset specification (HuggingFace dataset, tokenizer)

## TODO

- [ ] Implement actual Transformer blocks in `model.py`
- [ ] Complete sentence head extraction in `SentenceToSectionRouter`
- [ ] Implement full child/parent/neighbour logic in attention biases
- [ ] Add main training loop in `training.py`
- [ ] Implement masked language modeling / causal LM objectives
- [ ] Add evaluation metrics and downstream tasks
