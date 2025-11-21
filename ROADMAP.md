# LHT Implementation Roadmap

This document tracks the current state of the Learned Hierarchical Transformer implementation and provides a prioritized plan to reach a fully functional pretraining system.

## Current State: ~60% Complete

The repository has a **solid architectural foundation** with flexible, config-driven design. All major abstractions are in place, but core components need implementation.

---

## ✅ What's Complete

### 1. Project Infrastructure (100%)
- ✅ `pyproject.toml` with all dependencies
- ✅ `uv` package management setup
- ✅ Pre-commit hooks (black, isort, ruff)
- ✅ CLI entrypoints: `lht-train-pretrain`, `lht-eval`
- ✅ Config system with YAML loading

### 2. Data Pipeline (100%)
- ✅ Multi-dataset loading (HUPD, unarXive, Wikipedia)
- ✅ Interleaved sampling with configurable probabilities
- ✅ HuggingFace tokenizer integration
- ✅ Streaming support for large datasets
- ✅ 8192-token sequence handling

### 3. Hierarchy System (80%)
- ✅ `HierarchyManager` with schedule-driven routing
- ✅ `LevelRouter` with STE for K levels
- ✅ Ratio loss computation per level
- ✅ Dynamic router placement via `at_layer`
- ✅ Statistics tracking (compression ratios, units per doc)
- ⚠️ **TODO**: Head extraction for level > 1 routers

### 4. Model Architecture (70%)
- ✅ Schedule-driven forward pass (no hard-coded stages)
- ✅ Dynamic attention bias selection
- ✅ Router schedule from config
- ✅ Single unified layer stack
- ❌ **MISSING**: Actual Transformer blocks
- ❌ **MISSING**: LM head for MLM predictions

### 5. Attention Masks (60%)
- ✅ Local sliding window bias
- ✅ Sentence-aware bias structure
- ✅ Generic hierarchical bias framework
- ⚠️ **TODO**: Parent/child edges implementation
- ⚠️ **TODO**: Sibling/neighbor logic
- ⚠️ **TODO**: [DOC] token connectivity

### 6. Training Utilities (70%)
- ✅ MLM masking (BERT-style: 80/10/10)
- ✅ MLM loss computation
- ✅ Training step skeleton
- ✅ Router loss integration
- ❌ **MISSING**: Optimizer setup
- ❌ **MISSING**: Learning rate scheduler
- ❌ **MISSING**: Gradient accumulation loop
- ❌ **MISSING**: Main training loop

### 7. Evaluation (10%)
- ✅ CLI stub
- ❌ **MISSING**: Perplexity computation
- ❌ **MISSING**: Downstream task evaluation

---

## 🎯 Priority Roadmap

### **Priority 1: Make It Run** (Core Functionality)

These are blocking issues that prevent training from starting.

#### 1.1 Implement Transformer Blocks ⭐⭐⭐
**File**: `src/lht/model.py`

```python
class LHTTransformerBlock(nn.Module):
    """Single transformer layer with xFormers attention."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

        self.attention = ... # xFormers memory_efficient_attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias, attention_mask):
        # Pre-norm transformer
        # Attention with bias
        # FFN
        # Residual connections
        pass
```

**Impact**: Unblocks all forward passes.

---

#### 1.2 Add MLM Head ⭐⭐⭐
**File**: `src/lht/model.py`

```python
class LHTEncoder(nn.Module):
    def __init__(self, config):
        # ... existing code ...

        # LM head (tied weights)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def forward(self, input_ids, attention_mask):
        # ... existing forward pass ...

        # Add LM head
        logits = self.lm_head(x)  # [B, N, vocab_size]

        return {
            "hidden": x,
            "logits": logits,  # NEW
            "hierarchy": hier_state,
            "router_ratio_loss": router_ratio_loss,
        }
```

**Impact**: Enables MLM loss computation.

---

#### 1.3 Implement Main Training Loop ⭐⭐⭐
**File**: `src/lht/main.py`

```python
def cli_train_pretrain(config: str):
    cfg = load_config(config)
    set_seed(cfg.seed)

    # Load data
    dataset, tokenizer = load_lht_pretrain_dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size)

    # Build model
    model = LHTEncoder(cfg.model).to(cfg.device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=cfg.training.num_steps,
    )

    # Training loop
    model.train()
    step = 0
    for batch in dataloader:
        result = training_step(model, batch, tokenizer, cfg)

        loss = result["loss"] / cfg.training.grad_accum_steps
        loss.backward()

        if (step + 1) % cfg.training.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging, checkpointing...
        step += 1
        if step >= cfg.training.num_steps:
            break
```

**Impact**: Unblocks training experiments.

---

### **Priority 2: Complete Hierarchy Logic** (Core Quality)

These improve the quality of learned hierarchies.

#### 2.1 Finish Attention Bias Logic ⭐⭐
**File**: `src/lht/attention_masks.py`

Implement in `HierarchicalBias.materialize()`:
- Parent edges: token → head at each level
- Sibling edges: units sharing parent at level+1
- Neighbor edges: ±k group indices at each level
- [DOC] token: bidirectional connectivity

**Impact**: Full HDT-style attention patterns.

---

#### 2.2 Implement Head Extraction ⭐⭐
**File**: `src/lht/routers.py`

```python
class LevelRouter(nn.Module):
    def forward(self, hidden, prev_level_ids, prev_is_heads, ...):
        if prev_is_heads is not None:
            # Extract and pool head representations
            head_mask = prev_is_heads > 0.5  # [B, N]
            # Pool within each prev-level unit
            # Run router on pooled reps
            # Broadcast decisions back to token space
        else:
            # Level 1: route over all tokens
            pass
```

**Impact**: Routers operate on appropriate granularity.

---

### **Priority 3: Production Features** (Robustness)

These make the system production-ready.

#### 3.1 Checkpointing & Resumption ⭐
**Files**: `src/lht/utils.py`, `src/lht/main.py`

- Save model state, optimizer state, scheduler state
- Track global step, best metrics
- Resume from checkpoint with validation

---

#### 3.2 Logging & Monitoring ⭐
**File**: `src/lht/main.py`

- Integrate wandb or tensorboard
- Log: MLM loss, router losses, compression stats
- Track: tokens/sec, memory usage
- Visualize: learned boundaries on sample docs

---

#### 3.3 Evaluation Framework ⭐
**Files**: `src/lht/main.py`, `src/lht/evaluation.py` (new)

- MLM perplexity on held-out data
- Boundary quality metrics (if gold available)
- Downstream task evaluation (sequence classification, etc.)

---

### **Priority 4: Research Experiments** (Science)

These enable the research questions.

#### 4.1 Ablation Suite
- 2 vs 3 vs 4 hierarchy levels
- Different router placement schedules
- Window sizes and neighbor counts
- With/without [DOC] token

#### 4.2 Analysis Tools
- Visualize learned boundaries
- Measure boundary consistency
- Compare to gold boundaries (when available)
- Attention pattern visualization

---

## 📊 Current Completeness

```
Overall:              [████████░░] 60%

Infrastructure:       [██████████] 100%
Data Pipeline:        [██████████] 100%
Hierarchy System:     [████████░░]  80%
Model Architecture:   [███████░░░]  70%
Training Loop:        [███████░░░]  70%
Attention Logic:      [██████░░░░]  60%
Evaluation:           [█░░░░░░░░░]  10%
```

---

## 🚀 Quick Start Path

To get a **minimal working system** in order:

1. **Day 1**: Implement Transformer blocks (1.1)
2. **Day 1**: Add LM head (1.2)
3. **Day 2**: Basic training loop (1.3) - no fancy features
4. **Day 2**: Test on small data (100 steps)
5. **Day 3**: Complete attention bias (2.1)
6. **Day 4**: Add checkpointing (3.1) + logging (3.2)
7. **Day 5+**: Full pretraining run

After this, you have a working LHT that can be trained and evaluated!

---

## 📝 Notes

### Design Decisions to Make

1. **RoPE vs Absolute PE**: Config has `rope: true` but not implemented
2. **Flash Attention**: xFormers vs torch.nn.functional.scaled_dot_product_attention
3. **Mixed Precision**: Config specifies bf16, needs AMP integration
4. **Gradient Checkpointing**: For memory efficiency with 8192 tokens

### Known Limitations

1. **Transformer blocks**: Currently empty, blocks forward pass
2. **Head extraction**: All routers see full token hidden states (not pooled heads)
3. **Attention masks**: Only same-unit edges implemented
4. **Training loop**: Just a skeleton with no optimizer

### Testing Strategy

1. **Unit tests**: Router boundary decisions, attention mask correctness
2. **Integration tests**: Full forward pass with fake data
3. **Smoke tests**: 100-step training run on tiny data
4. **Full experiments**: HDT-comparable 10k steps on real data

---

## 🎓 Resources for Implementation

- **xFormers docs**: https://facebookresearch.github.io/xformers/
- **HDT paper**: For attention pattern details
- **H-Net paper**: For router architecture inspiration
- **BERT repo**: For MLM implementation reference
- **Cramming repo**: For training loop patterns

---

This roadmap will be updated as tasks are completed. Current focus: **Priority 1** to get a minimal working system.
