# Learned Hierarchical Transformer (LHT)

A hierarchical transformer encoder implementation with **variable-depth learned hierarchies**:
- **H-Net-style routing**: Learn K abstraction levels without supervision
- **HDT-style hierarchical attention**: Child/parent/sibling attention patterns via xFormers
- **Multi-scale processing**: Local → Mid → Global layers with different attention patterns
- **Flexible hierarchy depth**: Support for 2, 3, 4, or N abstraction levels

## Variable-Depth Learned Hierarchies

Unlike HDT which fixes 2 linguistic levels (sentences and sections), LHT supports arbitrary depth with **learned** boundaries:

| K Levels | Hierarchy Structure |
|----------|-------------------|
| **2** | tokens → level-0 groups |
| **3** | tokens → level-0 → level-1 *(default)* |
| **4** | tokens → level-0 → level-1 → level-2 |
| **N** | arbitrary depth |

All boundaries are learned via H-Net routers (no linguistic supervision). This is controlled via simple config changes - no code modifications required!

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
2. **Router fires after layer 3**: Learn level-0 boundaries (finest-grained groups)
3. **Layers 4-6**: Level-0 hierarchical attention
4. **Router fires after layer 6**: Learn level-1 boundaries (coarser groups)
5. **Layers 7-12**: Full K=2 hierarchical attention (level-0 + level-1)

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
- Works for any level: tokens→level-0, level-0→level-1, level-1→level-2, etc.

**Why no upsampling module?**

H-Net includes an upsampling module because it's an autoencoder (encoder compresses → decoder reconstructs). LHT is a hierarchical transformer with progressive abstraction across K learned levels, not compression-reconstruction. Higher hierarchy levels intentionally don't need token-level resolution, and upsampling would reintroduce O(N²) attention costs. Therefore, LHT uses only H-Net's dynamic chunking and ratio loss, not the autoencoder-specific upsampling/smoothing modules.

**Why hybrid HDT + neighbor windows?**

HDT assumes all levels are known from the start (sentences, sections, document), so siblings = "same parent". LHT learns levels progressively:
- After Router 1: Only level 0 exists (no parent yet)
- After Router 2: Levels 0-1 exist (level 0 has parent, level 1 doesn't)
- After Router K: All K levels exist

The **top level at any point has no parent yet**, so we use a numeric neighbor window as fallback. Once the next router creates a parent level, we automatically switch to HDT siblings (same parent = all neighbors). This hybrid approach:
- **With parent (ℓ+1 exists)**: Pure HDT — e.g., all 20 sentence heads in same section communicate
- **Without parent (top level)**: Numeric window — e.g., level-1 heads attend to ±1 neighbors
- **Progressive**: Transitions from window → HDT as hierarchy grows

### Attention Biases

- `build_local_attention_bias()`: Sliding window for token-level (no hierarchy yet)
- `HierarchicalBias`: Generic K-level child/parent/sibling attention with **hybrid HDT + fallback**
  - Level 0: all tokens in same group
  - Level ℓ>0: previous-level heads within group
  - **Sibling edges (hybrid)**:
    - If level ℓ+1 exists: HDT siblings (same parent = all neighbors in group)
    - If level ℓ is top: numeric neighbor window (fallback until parent learned)
  - [DOC] token connecting to top-level heads only

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
- `attention`: Attention patterns (local window, neighbor fallback, DOC token)
- `training`: Optimization settings (batch_size, learning_rate, steps, MLM probability)
- `data`: Dataset specification (multiple sources, sampling probabilities)

### Configuring Hierarchy Depth & Schedule

The `at_layer` field controls when each router fires. This determines the attention pattern transitions:

**K=1 (single learned level) with local layers 1-6:**
```yaml
model:
  num_layers: 12
hierarchy:
  levels:
    - name: "level0"
      at_layer: 6              # router fires after layer 6
      target_head_ratio: 0.03
      loss_weight: 0.05
attention:
  neighbour_windows: [1]       # Level 0 is top (no parent), uses window fallback
```

**K=2 (default, two learned levels):**
```yaml
hierarchy:
  levels:
    - name: "level0"
      at_layer: 3              # local → level-0 hierarchy transition
      target_head_ratio: 0.03
      loss_weight: 0.05
    - name: "level1"
      at_layer: 6              # level-0 → full K=2 hierarchy transition
      target_head_ratio: 0.15
      loss_weight: 0.05
attention:
  neighbour_windows: [1, 1]    # Level 0 uses HDT (has parent), Level 1 uses window (top)
```

**K=3 (three learned levels):**
```yaml
hierarchy:
  levels:
    - name: "level0"
      at_layer: 3
      target_head_ratio: 0.03
      loss_weight: 0.05
    - name: "level1"
      at_layer: 6
      target_head_ratio: 0.08
      loss_weight: 0.05
    - name: "level2"
      at_layer: 9
      target_head_ratio: 0.15
      loss_weight: 0.05
attention:
  neighbour_windows: [1, 1, 2]  # Levels 0-1 use HDT, Level 2 (top) uses wider window
```

**Shallow model (6 layers, K=1):**
```yaml
model:
  num_layers: 6
hierarchy:
  levels:
    - name: "level0"
      at_layer: 3
      target_head_ratio: 0.03
      loss_weight: 0.05
```

## Key Differences from HDT

While LHT aligns with HDT's training setup for fair comparison, it differs in several key ways:

| Aspect | HDT | LHT |
|--------|-----|-----|
| **Hierarchy** | Fixed linguistic (sentences/sections) | Learned via H-Net routers |
| **Depth** | Fixed 2 levels | Variable K levels (1, 2, 3, or more) |
| **Architecture** | Fixed local/mid/global stages | Schedule-driven (configurable) |
| **Preprocessing** | Requires NLTK sentence splitting | Raw text only |
| **Boundaries** | Explicit [SEC] tokens | Learned boundaries (cosine similarity + threshold) |
| **Supervision** | Linguistic structure required | Fully unsupervised |
| **Router Placement** | N/A | Configurable via `at_layer` |

## TODO

- [x] Implement Transformer blocks with xFormers memory-efficient attention
- [x] Add MLM head with weight tying
- [x] Implement generic K-level hierarchical attention with strict HDT sparsity
- [x] Add H-Net routing with cosine similarity and ratio loss
- [x] Implement full training loop with W&B logging
- [ ] Implement data collator for dynamic batching
- [ ] Add evaluation metrics and downstream tasks
- [ ] Add support for variable sequence lengths within batch
- [ ] Add checkpointing and resumption logic
- [ ] Add logging with wandb/tensorboard
