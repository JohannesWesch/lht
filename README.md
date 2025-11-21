# Learned Hierarchical Transformer (LHT)

A hierarchical transformer encoder implementation with **variable-depth learned hierarchies**:
- **H-Net-style routing**: Learn K abstraction levels without supervision
- **HDT-style hierarchical attention**: Child/parent/sibling attention patterns via xFormers
- **Multi-scale processing**: Local → Mid → Global layers with different attention patterns
- **Flexible hierarchy depth**: Support for 2, 3, 4, or N abstraction levels

## Variable-Depth Hierarchies

Unlike HDT which fixes 2 levels (sentences and sections), LHT supports arbitrary hierarchy depth:

| K Levels | Hierarchy Structure |
|----------|-------------------|
| **2** | tokens → sentences |
| **3** | tokens → sentences → sections *(default)* |
| **4** | tokens → sentences → paragraphs → sections |
| **N** | arbitrary depth |

This is controlled via a simple config change - no code modifications required!

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

LHT uses a **schedule-driven architecture** instead of hard-coded stages:

- **Single layer stack**: One unified list of transformer layers (no fixed local/mid/global separation)
- **Dynamic attention bias**: Chosen based on how many hierarchy levels exist at each layer
- **Configurable router placement**: Each router fires after a specified layer (via `at_layer`)

This makes the architecture fully flexible:
- Want local layers 1-3, then hierarchical? Set router `at_layer: 3`
- Want to add a third level after layer 9? Just add it to config
- Want only 6 layers total? Change `num_layers: 6`

**Example schedule (default):**
1. **Layers 1-3**: Sliding window attention (before any routers)
2. **Router fires after layer 3**: Learn sentence boundaries
3. **Layers 4-6**: Sentence-aware attention
4. **Router fires after layer 6**: Learn section boundaries
5. **Layers 7-12**: Full hierarchical attention (sentences + sections)

### Hierarchy System

**`HierarchyManager`**: Central component managing K abstraction levels

- Creates K routers automatically from config
- Runs them sequentially to build full hierarchy
- Maintains `level_ids[ℓ]` and `is_heads[ℓ]` for each level ℓ = 1..K
- Computes ratio losses for all levels
- Provides compression statistics

**`LevelRouter`**: Generic router for any abstraction level (H-Net Dynamic Chunking)

- Implements H-Net's dynamic chunking mechanism (https://arxiv.org/abs/2507.07955)
- **Cosine similarity** between adjacent projected states (Eq. 4)
- **Hard boundary indicators** via threshold: `b_t = 1[p_t >= 0.5]` (no Gumbel-Softmax)
- **Ratio loss** to encourage target compression factor (Eq. 10)
- Works for: token→sentence, sentence→paragraph, paragraph→section, etc.

**Why no upsampling module?**

H-Net includes an upsampling module because it's an autoencoder (encoder compresses → decoder reconstructs). LHT is a hierarchical transformer with progressive abstraction (tokens → sentences → sections), not compression-reconstruction. Higher hierarchy levels intentionally don't need token-level resolution, and upsampling would reintroduce O(N²) attention costs. Therefore, LHT uses only H-Net's dynamic chunking and ratio loss, not the autoencoder-specific upsampling/smoothing modules.

### Attention Biases

- `build_local_attention_bias()`: Sliding window for local layers
- `SentenceAwareBias`: Restricts attention to first-level units + heads
- `HierarchicalBias`: Generic K-level child/parent/neighbour attention

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

See `configs/pretrain_base.yaml` for a full example. The default config is aligned with **HDT (Hierarchical Document Transformer)** pretraining setup:

### HDT Alignment

LHT's default configuration matches HDT-E (encoder-only) for fair comparison:

**Model Architecture:**
- 12 layers (3 local + 3 mid + 6 global)
- 768 hidden dim, 12 attention heads
- 3072 FFN dim
- 8192 token sequence length
- ~109M parameters (BERT-base size)

**Training Setup:**
- **Objective:** Masked Language Modeling (MLM)
- **Masking:** 15% of tokens (80% [MASK], 10% random, 10% unchanged)
- **Global Batch Size:** 128 sequences (via gradient accumulation)
- **Steps:** 10,000 (~24h on 1 GPU)
- **Learning Rate:** 3e-4 with warmup
- **Sequence Length:** 8192 tokens

**Data:**
- **Corpora:** unarXive, HUPD, Wikipedia (~12M documents)
- **Sampling:** 40% unarXive, 30% HUPD, 30% Wikipedia
- **Tokenizer:** BERT-base-uncased (or custom BPE vocab=32768)

### Config Sections

- `model`: Architecture hyperparameters (d_model, n_heads, layer counts)
- `hierarchy`: **Variable-depth hierarchy configuration**
  - `levels`: List of abstraction levels (controls K)
  - `router`: Shared router architecture for all levels
- `attention`: Attention patterns (window sizes, neighbour counts)
- `training`: Optimization settings (batch_size, learning_rate, steps, MLM probability)
- `data`: Dataset specification (multiple sources, sampling probabilities)

### Configuring Hierarchy Depth & Schedule

The `at_layer` field controls when each router fires. This determines the "local/mid/global" behavior:

**2 levels (sentences only) with local layers 1-6:**
```yaml
model:
  num_layers: 12
hierarchy:
  levels:
    - name: "sentence"
      at_layer: 6              # router fires after layer 6
      target_head_ratio: 0.03
      loss_weight: 0.05
```

**3 levels (default) with local/sentence-aware/hierarchical:**
```yaml
hierarchy:
  levels:
    - name: "sentence"
      at_layer: 3              # local → sentence-aware transition
      target_head_ratio: 0.03
      loss_weight: 0.05
    - name: "section"
      at_layer: 6              # sentence-aware → hierarchical transition
      target_head_ratio: 0.15
      loss_weight: 0.05
```

**4 levels (add paragraphs):**
```yaml
hierarchy:
  levels:
    - name: "sentence"
      at_layer: 3
      target_head_ratio: 0.03
      loss_weight: 0.05
    - name: "paragraph"
      at_layer: 6
      target_head_ratio: 0.08
      loss_weight: 0.05
    - name: "section"
      at_layer: 9
      target_head_ratio: 0.15
      loss_weight: 0.05
```

**Shallow model (6 layers, 1 level):**
```yaml
model:
  num_layers: 6
hierarchy:
  levels:
    - name: "sentence"
      at_layer: 3
      target_head_ratio: 0.03
      loss_weight: 0.05
```

## Key Differences from HDT

While LHT aligns with HDT's training setup for fair comparison, it differs in several key ways:

| Aspect | HDT | LHT |
|--------|-----|-----|
| **Hierarchy** | Fixed (gold sentences/sections) | Learned (H-Net-style routers) |
| **Depth** | Fixed 2 levels | Variable K levels (2, 3, 4, or more) |
| **Architecture** | Fixed local/mid/global stages | Schedule-driven (configurable) |
| **Preprocessing** | Requires NLTK sentence splitting | Raw text only |
| **Boundaries** | Explicit [SEC] tokens | Soft router probabilities + STE |
| **Supervision** | Structured input required | Fully unsupervised |
| **Router Placement** | N/A | Configurable via `at_layer` |

## TODO

- [ ] Implement actual Transformer blocks in `model.py`
- [ ] Add LM head for MLM predictions
- [ ] Complete sentence head extraction in `SentenceToSectionRouter`
- [ ] Implement full child/parent/neighbour logic in attention biases
- [ ] Add main training loop with optimizer and scheduler
- [ ] Implement data collator for dynamic batching
- [ ] Add evaluation metrics and downstream tasks
- [ ] Add checkpointing and resumption logic
- [ ] Add logging with wandb/tensorboard
