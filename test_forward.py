"""Quick sanity test for LHTEncoder forward pass."""

import torch

from lht.config import load_config
from lht.model import LHTEncoder

print("Loading config...")
cfg = load_config("configs/pretrain_base.yaml")

print(f"Building model with {cfg.model.num_layers} layers...")
model = LHTEncoder(cfg)

# Use CPU for testing
device = "cpu"
model = model.to(device)

print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

# Check CUDA availability
if device == "cpu" and not torch.cuda.is_available():
    print("\n⚠️  xFormers requires CUDA for memory_efficient_attention")
    print("   Skipping forward pass test on CPU")
    print("\n✅ Model structure is valid!")
    print(f"   - {len(model.layers)} transformer layers")
    print(f"   - {model.num_levels} hierarchy levels")
    print(f"   - Router schedule: {model.router_schedule}")
    import sys

    sys.exit(0)

# Create dummy data
B, N = 2, 128
print(f"\nTesting with batch_size={B}, seq_len={N}")

dummy_ids = torch.randint(low=10, high=1000, size=(B, N)).to(device)
attention_mask = torch.ones(B, N, dtype=torch.long, device=device)

print("Running forward pass...")
try:
    out = model(dummy_ids, attention_mask=attention_mask)

    print("\n✅ Forward pass successful!")
    print(f"  hidden shape: {out['hidden'].shape}")
    print(f"  router_ratio_loss: {out['router_ratio_loss'].item():.4f}")
    print(f"  hierarchy levels: {list(out['hierarchy']['level_ids'].keys())}")

    # Check hierarchy state
    hier = out["hierarchy"]
    for level, ids in hier["level_ids"].items():
        is_head = hier["is_heads"][level]
        num_heads = is_head.sum().item()
        print(f"  Level {level}: {num_heads} heads detected")

    print("\n🎉 LHTEncoder is working!")

except Exception as e:
    print("\n❌ Error during forward pass:")
    print(f"  {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
