# Visualization Guide for ML-SWA Training

This guide explains how to use visualizations during training and for debugging.

## Overview

The visualization system provides:

1. **Attention Heatmaps** - Visualize ML-SWA attention patterns
2. **MLM Prediction Tables** - Track model predictions during training
3. **Gradient Flow Charts** - Monitor gradient health
4. **Attention Entropy** - Detect attention collapse/diffusion

## Quick Start

### 1. Training with Visualizations

The main training script (`scripts/train_pretrain.py`) automatically includes visualization callbacks when W&B is enabled:

```bash
# Start training with visualizations
uv run python scripts/train_pretrain.py --config configs/pretrain_hierarchical.yaml
```

Visualizations will be logged to W&B every 1000 steps by default.

### 2. Single Batch Visualization (for debugging)

To quickly visualize attention on a sample document:

```bash
# With a config file (random weights)
uv run python tools/visualize_batch.py --config configs/pretrain_hierarchical.yaml

# With a trained checkpoint
uv run python tools/visualize_batch.py --checkpoint checkpoints/my-model/checkpoint-10000.ckpt

# Specify output location
uv run python tools/visualize_batch.py \
    --config configs/pretrain_hierarchical.yaml \
    --output my_attention_heatmap.png \
    --max-tokens 80
```

This will:
- Create a sample hierarchical document
- Run forward pass
- Generate attention heatmap
- Compute entropy statistics
- Save visualization to file

## Configuration

### Control Visualization Frequency

Add to your config YAML:

```yaml
training:
  log_viz_every: 1000  # Log visualizations every N steps (default: 1000)
```

### Disable Visualizations

Visualizations are automatically disabled when W&B is in offline mode:

```yaml
wandb:
  offline: true  # No visualizations
```

## What Each Visualization Shows

### 1. Attention Heatmap

**What it shows:** Token-to-token attention weights for the first layer

**How to interpret:**
- **Diagonal bands:** Local token-level sliding window (±10 positions)
- **Cross-patterns:** Sentence boundary connections
- **Bright intersections:** Section boundary long-range attention
- **Color intensity:** Attention weight strength (darker = stronger)

**Logged to W&B as:** `attention/layer_0`

### 2. MLM Prediction Table

**What it shows:** Model predictions for masked tokens

**Columns:**
- `Sample`: Batch index
- `Position`: Token position
- `Original`: The actual token (before masking)
- `Masked`: What the model saw ([MASK] token)
- `Top-5 Predictions`: Model's top 5 predictions with probabilities
- `Correct`: ✓ if top prediction matches original, ✗ otherwise

**Logged to W&B as:** `predictions/step_{N}`

### 3. Gradient Flow Chart

**What it shows:** Gradient norms for each layer

**Colors:**
- **Blue bars:** Normal gradients
- **Red bars:** Vanishing gradients (< 1e-5) ⚠️
- **Orange bars:** Exploding gradients (> 10) ⚠️

**Logged to W&B as:** `gradients/flow`

### 4. Attention Entropy

**What it shows:** How focused vs diffuse attention is

**Metrics:**
- `attention/entropy_mean`: Average entropy across all positions
- `attention/entropy_std`: Entropy variance
- `attention/entropy_min`: Most focused attention
- `attention/entropy_max`: Most diffuse attention

**Interpretation:**
- **Low entropy (< 1.0):** Very focused attention (good for specific tasks)
- **High entropy (> 2.0):** Very diffuse attention (may indicate issues)
- **Normal range:** 1.5-2.5 for most transformers

## Enabling Attention Weight Returns

⚠️ **Important:** By default, the model does NOT return attention weights (for speed).

To enable attention visualization, modify `src/lht/model.py`:

```python
def forward(self, input_ids, attention_mask=None, positions=None):
    # ... existing code ...

    # Store attention weights from each layer
    attention_weights = []
    for layer in self.layers:
        x, attn = layer(x, positions=positions, return_attention=True)
        attention_weights.append(attn)

    # ... rest of code ...

    return {
        "mlm_logits": logits,
        "hidden": x,
        "attentions": attention_weights,  # Add this line
    }
```

Then modify `src/lht/core/attention.py` to optionally return attention weights:

```python
def mlswa_attention(query, key, value, positions, window_size_per_level, return_attention=False):
    # ... existing code ...

    attn_output = _compiled_mlswa_attention(query_t, key_t, value_t, block_mask=block_mask)

    # Transpose back
    attn_output = attn_output.transpose(1, 2)  # [B, N, H, D]

    if return_attention:
        # Compute attention weights for visualization
        # (This is expensive, only do when needed)
        scores = torch.matmul(query_t, key_t.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        return attn_output, attn_weights

    return attn_output
```

## Example W&B Dashboard

After training starts, check your W&B dashboard:

```
https://wandb.ai/{your-entity}/{lht-training}/{run-name}
```

You'll see:
- **Media** tab: Attention heatmaps and gradient flow charts
- **Tables** tab: MLM prediction tables
- **Charts** tab: Entropy trends over time

## Troubleshooting

### No visualizations appearing

1. **Check W&B is enabled:**
   ```yaml
   wandb:
     offline: false
   ```

2. **Check you've reached log_viz_every steps:**
   - Default is 1000 steps
   - First visualization appears at step 1000

3. **Check console output:**
   - Should see: `✓ Visualization callbacks enabled (logging every 1000 steps)`

### "Model does not return attention weights"

This is expected! To enable:
1. Modify your model to return attention weights (see "Enabling Attention Weight Returns" above)
2. MLM predictions and gradient flow will still work without this

### CUDA errors with tools/visualize_batch.py

The visualization tool works on CPU:
```bash
# Run on CPU
CUDA_VISIBLE_DEVICES="" uv run python tools/visualize_batch.py --config configs/pretrain_hierarchical.yaml
```

## Best Practices

1. **During development:** Use `tools/visualize_batch.py` frequently to check attention patterns
2. **During training:** Let callbacks handle automatic logging
3. **For papers:** Use `tools/visualize_batch.py` with trained checkpoints to generate publication-quality figures
4. **Performance:** Only enable attention weight returns when you need visualizations (adds ~10-20% overhead)

## Related Files

- **Main training:** `scripts/train_pretrain.py`
- **Visualization callbacks:** `src/lht/callbacks.py`
- **Visualization functions:** `src/lht/visualization.py`
- **Debug tool:** `tools/visualize_batch.py`
- **Example demo:** `demo_hierarchical_attention.py`

## Questions?

Check the example demo:
```bash
uv run python demo_hierarchical_attention.py
```

This creates a complete example with sample document, hierarchical positions, simulated attention, and visualization.
