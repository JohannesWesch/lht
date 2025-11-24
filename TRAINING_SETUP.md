# LHT Training Setup Documentation

## Overview

The Learned Hierarchical Transformer (LHT) is trained for Masked Language Modeling (MLM) on long documents with hierarchical structure. The key innovation is **geometric attention** that uses Manhattan distance in a 2D coordinate space to enforce hierarchical parent-child relationships while applying per-level sliding windows.

---

## 1. Data Loading and Hierarchical Structure

### 1.1 Dataset Format

**Source**: Streaming datasets from HuggingFace Hub
- Primary: `howey/wiki_en` (Wikipedia articles in nested format)
- Configuration: `configs/pretrain_hierarchical.yaml` lines 64-67

**Key characteristic**: Data is structured as **nested lists** representing document hierarchy:

```python
document = [
    ["Sentence 1 in section 1.", "Sentence 2 in section 1."],  # Section 1
    ["Sentence 1 in section 2.", "Sentence 2 in section 2."],  # Section 2
    # ... more sections
]
```

### 1.2 Hierarchy Levels

The model uses **3 levels** of hierarchy (config line 26):
- **Level 0**: Individual tokens (words/subwords)
- **Level 1**: Sentences (second-level nesting)
- **Level 2**: Sections (top-level nesting)

### 1.3 Coordinate Building Process

**File**: `src/lht/utils/nested_builder.py`

The `build_coords_from_nested_list()` function converts nested documents into:
1. **Flat token sequence** (input_ids): All tokens concatenated, max 8192 tokens
2. **GeometricCoordinates**: 2D (x, y) coordinates for each token

**Coordinate assignment logic** (`src/lht/core/coords.py`):
- **y-coordinate (level)**: Hierarchy level (0, 1, or 2)
- **x-coordinate (logical_time)**: Parent group ID

**Example**:
```
Document: [["Hello world", "How are you"], ["Fine thanks"]]

Tokens: [Hello, world, How, are, you, Fine, thanks]
         ├─────────────┤ └───────────┤  └─────────┤
         Sentence 0     Sentence 1     Sentence 2
         └──────────────────────────┤  └─────────┤
                Section 0               Section 1

Coordinates:
Token "Hello":  (x=0, y=0) - level 0, parent sentence_id=0
Token "world":  (x=0, y=0) - level 0, parent sentence_id=0
Token "How":    (x=1, y=0) - level 0, parent sentence_id=1
Token "are":    (x=1, y=0) - level 0, parent sentence_id=1
Token "you":    (x=1, y=0) - level 0, parent sentence_id=1
Token "Fine":   (x=2, y=0) - level 0, parent sentence_id=2
Token "thanks": (x=2, y=0) - level 0, parent sentence_id=2

Virtual "Sentence 0": (x=0, y=1) - level 1, parent section_id=0
Virtual "Sentence 1": (x=1, y=1) - level 1, parent section_id=0
Virtual "Sentence 2": (x=2, y=1) - level 1, parent section_id=1
Virtual "Section 0":  (x=0, y=2) - level 2, self-parent
Virtual "Section 1":  (x=1, y=2) - level 2, self-parent
```

**Key insight**: Children of the same parent share the same x-coordinate, ensuring Manhattan distance = 1 for parent-child pairs.

### 1.4 Data Collation

**File**: `src/lht/data/mlm.py` - `HierarchicalDataCollator`

For each batch:
1. Build coordinates for each document
2. Tokenize to get `input_ids`
3. Pad sequences to same length (lines 53-55)
4. Pad/truncate coordinate tensors to match (lines 69-78)
5. Apply MLM masking (15% of tokens → `[MASK]`)
6. Stack into batch with `GeometricCoordinates` object

---

## 2. Geometric Attention Mechanism

### 2.1 Manhattan Distance Formula

**File**: `src/lht/core/attention.py` - `geometric_mask_fn()` (lines 93-137)

For any query token at position `q` and key token at position `k`:

```
Manhattan distance = |q.level - k.level| + |q.logical_time - k.logical_time|
                   = |Δy| + |Δx|
```

**Attention rule**: Query `q` can attend to key `k` if:
```
Manhattan distance ≤ radius
```

**Configuration**: `manhattan_radius: 1` (config line 27)

### 2.2 Why Manhattan Distance = 1?

With radius = 1, a token can attend to:
1. **Siblings**: Same parent (Δx=0, Δy=0) → distance = 0 ✅
2. **Parent**: One level up, same x (Δx=0, Δy=1) → distance = 1 ✅
3. **Children**: One level down, same x (Δx=0, Δy=-1) → distance = 1 ✅
4. **Uncles/cousins**: Different parent (Δx≥1) → distance > 1 ❌

This creates **structured attention patterns**:
- Tokens within same sentence attend to each other
- Tokens attend to their parent sentence summary
- Sentence summaries attend to their section
- Prevents cross-section information leakage (except through shared parents)

### 2.3 Implementation Details

**Core function**: `geometric_attention()` (lines 162-208)

1. **Mask creation** (`create_geometric_mask`, lines 48-145):
   - Uses PyTorch's `create_block_mask` API
   - Mask function is called for each (q_idx, kv_idx) pair
   - Returns boolean: True if attention allowed

2. **FlexAttention** (line 203):
   - Efficient sparse attention implementation
   - Format: `[B, H, N, D]` (batch, heads, seq_len, head_dim)
   - Compiled with `torch.compile()` for performance (line 159)

3. **Tensor shape handling**:
   - Input: `[B, N, H, D]` (batch, seq, heads, dim)
   - Transpose to `[B, H, N, D]` for FlexAttention (lines 188-190)
   - Transpose back to `[B, N, H, D]` for output (line 206)

---

## 3. Sliding Window Mechanism

### 3.1 Per-Level Window Sizes

**Configuration** (config line 28):
```yaml
window_size_per_level: [256, 64, 16]
#                       ^^^  ^^  ^^
#                       L0   L1  L2
#                     tokens sent sect
```

### 3.2 Window Constraint Logic

**File**: `src/lht/core/attention.py` - `geometric_mask_fn()` (lines 125-135)

For tokens at the **same level** (same y-coordinate):
```
Can attend if: |q.logical_time - k.logical_time| ≤ window_size[level]
```

**Example at Level 0 (tokens)**:
- Window size = 256 tokens
- Token at position 100 can attend to tokens 0-356 in same sentence
- This prevents quadratic attention over entire 8192 sequence

**Example at Level 1 (sentences)**:
- Window size = 64 sentences
- A sentence can attend to 64 neighboring sentences in same section

**Implementation** (lines 126-135):
```python
same_level = q_level == k_level
within_window = dist_time <= window_sizes[q_level]
# Apply window ONLY for same-level pairs
within_radius = within_radius & (~same_level | within_window)
```

**Key**: Window constraint applies only to same-level attention; cross-level (parent-child) attention is still governed by Manhattan distance.

---

## 4. Model Architecture

### 4.1 LHTEncoder Structure

**File**: `src/lht/model.py`

**Configuration** (matching HDT-E from paper):
- `d_model: 768` (hidden dimension)
- `n_heads: 12` (attention heads)
- `num_layers: 12` (transformer blocks)
- `d_ff: 3072` (feedforward dimension, 4 × d_model)
- `vocab_size: 30522` (BERT tokenizer)

**Components**:
1. **Token embeddings** (line 41): `nn.Embedding(vocab_size, d_model)`
   - Initialized with `N(0, 0.02)` (BERT-style)
2. **12 × GeometricTransformerBlock** (lines 46-63)
3. **Final LayerNorm** (line 68): Critical for stable logits
4. **MLM head** (line 71): Linear projection to vocabulary
   - No weight tying (for training stability)
   - Initialized with `N(0, 0.02)`

**Total parameters**: ~53.5M trainable

### 4.2 GeometricTransformerBlock

**File**: `src/lht/core/model.py`

Standard transformer architecture with geometric attention:
1. **Pre-norm + Geometric Multi-Head Attention** (lines 76-105)
   - Uses `geometric_attention()` instead of standard attention
   - Passes coordinates and window sizes
2. **Residual connection** (line 110)
3. **Pre-norm + Feedforward (GELU)** (lines 113-117)
4. **Residual connection** (line 118)

**Per-layer level control** (config lines 29-41):
```yaml
layer_max_level: [2, 2, 2, ..., 2]  # All 12 layers use all 3 levels
```
This allows restricting lower layers to only token-level (0) or sentence-level (0-1) attention.

---

## 5. Training Configuration

### 5.1 Optimization

**File**: `configs/pretrain_hierarchical.yaml` (lines 43-56)

- **Batch size**: 2 per GPU
- **Gradient accumulation**: 16 steps → Effective batch size = 32
- **Learning rate**: 1e-4 (0.0001)
- **Warmup steps**: 1,000 (reduced from 10k to allow faster LR ramp-up)
- **Weight decay**: 0.01
- **Mixed precision**: `bf16-mixed` (bfloat16 for performance)
- **Max gradient norm**: 1.0 (gradient clipping)
- **Optimizer**: AdamW (default in PyTorch Lightning)

### 5.2 MLM Configuration

- **Masking probability**: 15% of tokens (line 56)
- **Tokenizer**: `google-bert/bert-base-uncased` (line 63)
  - Has `[MASK]` token required for MLM
- **Max sequence length**: 8192 tokens (line 62)

### 5.3 Training Loop

**File**: `src/lht/lightning_module.py`

1. **Forward pass** (lines 35-44):
   - Extract `input_ids`, `labels`, `coords` from batch
   - Call `model(input_ids, coords=coords)`
   - Get `mlm_logits` from output

2. **Loss computation** (`src/lht/training.py`):
   - `CrossEntropyLoss` over masked tokens only
   - Ignore index = -100 (unmasked tokens)

3. **Logging** (lines 48-55):
   - Train loss every 10 steps
   - Validation every 1000 steps (50 batches with streaming data)
   - Logged to Weights & Biases

### 5.4 Scheduler

**Linear warmup** from 0 → 1e-4 over 1000 steps, then constant LR.

---

## 6. Key Files Reference

| Component | File Path | Key Functions/Classes |
|-----------|-----------|----------------------|
| Data loading | `src/lht/data/mlm.py` | `HierarchicalDataCollator`, `MLMDataModule` |
| Coordinate builder | `src/lht/utils/nested_builder.py` | `build_coords_from_nested_list()` |
| Core coords | `src/lht/core/coords.py` | `build_coords()` |
| Geometric attention | `src/lht/core/attention.py` | `geometric_attention()`, `create_geometric_mask()` |
| Transformer block | `src/lht/core/model.py` | `GeometricTransformerBlock` |
| LHT model | `src/lht/model.py` | `LHTEncoder` |
| Lightning module | `src/lht/lightning_module.py` | `LHTLightningModule` |
| Training script | `scripts/train_pretrain.py` | Main entry point |
| Config | `configs/pretrain_hierarchical.yaml` | All hyperparameters |

---

## 7. Critical Implementation Details

### 7.1 Avoiding vmap Issues

The `geometric_mask_fn()` must use **scalar operations only** (no data-dependent control flow like `if`):
- Uses `torch.where()` instead of Python `if` statements
- Uses boolean operations (`&`, `|`, `~`) for combining conditions
- Required for `create_block_mask()` which uses `vmap` internally

### 7.2 Stability Features

1. **Final LayerNorm** before MLM head (model.py line 109)
   - Prevents logit explosion
   - Critical for keeping loss in reasonable range (~10)

2. **Proper weight initialization** (model.py lines 42, 72)
   - Embeddings and MLM head: `N(0, 0.02)`
   - Prevents extreme activations early in training

3. **No weight tying** (model.py line 71-72)
   - Embedding and MLM head have separate weights
   - Improves training stability for hierarchical model

### 7.3 Streaming Dataset Handling

**File**: `src/lht/data/basic.py`

- Detects `IterableDataset` instances
- Disables shuffling (not supported for streaming)
- Disables `DistributedSampler` (incompatible)
- Limits validation to 50 batches (streaming has no fixed end)

---

## Summary

The LHT training setup implements hierarchical document modeling through:

1. **Nested data structure** → Flattened tokens + 2D geometric coordinates
2. **Manhattan distance** (radius=1) → Parent-child attention only
3. **Sliding windows** per level → Prevents quadratic complexity
4. **FlexAttention** → Efficient sparse attention implementation
5. **MLM pretraining** → Learn contextual representations at all hierarchy levels

This architecture allows the model to process long documents (8192 tokens) while maintaining structured attention patterns that respect document hierarchy.
