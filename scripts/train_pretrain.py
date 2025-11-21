"""
Comprehensive pretraining script for LHT with W&B logging.

This script implements the full training loop with:
- Masked Language Modeling objective
- Router hierarchy learning
- Comprehensive W&B logging (metrics, gradients, visualizations)
- Mixed precision training
- Gradient accumulation
- Checkpointing
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lht.config import load_config
from lht.dataset import load_lht_pretrain_dataset
from lht.model import LHTEncoder
from lht.training import compute_mlm_loss, mask_tokens_for_mlm
from lht.utils import set_seed
from lht.visualization import (
    log_gradient_flow,
    log_router_statistics,
    log_sample_predictions,
)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and decay.

    Args:
        optimizer: optimizer
        num_warmup_steps: warmup steps
        num_training_steps: total training steps

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def compute_layer_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient norms for each layer.

    Args:
        model: LHT model

    Returns:
        dict mapping layer name to gradient norm
    """
    grad_norms = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[f"grad_norm/{name}"] = param.grad.norm().item()

    return grad_norms


def compute_layer_activation_stats(
    model: nn.Module, hidden_states: torch.Tensor, layer_idx: int
) -> Dict[str, float]:
    """
    Compute activation statistics for a layer.

    Args:
        model: LHT model
        hidden_states: [B, N, D] hidden states
        layer_idx: layer index

    Returns:
        dict of statistics
    """
    with torch.no_grad():
        stats = {
            f"activations/layer_{layer_idx}_mean": hidden_states.mean().item(),
            f"activations/layer_{layer_idx}_std": hidden_states.std().item(),
            f"activations/layer_{layer_idx}_max": hidden_states.max().item(),
            f"activations/layer_{layer_idx}_min": hidden_states.min().item(),
        }

    return stats


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    config,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """
    Save model checkpoint.

    Args:
        model: LHT model
        optimizer: optimizer
        scheduler: learning rate scheduler
        step: current training step
        config: experiment config
        checkpoint_dir: directory to save checkpoint
        is_best: whether this is the best checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint to {best_path}")

    # Upload to wandb if enabled
    if config.wandb.log_model:
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description=f"LHT model checkpoint at step {step}",
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)


def train_lht_pretrain(config_path: str, resume_from: Optional[str] = None):
    """
    Main training function for LHT pretraining.

    Args:
        config_path: path to YAML config file
        resume_from: optional path to checkpoint to resume from
    """
    # Load config
    print(f"Loading config from {config_path}...")
    cfg = load_config(config_path)

    # Set seed for reproducibility
    set_seed(cfg.seed)

    # Initialize wandb
    run_name = cfg.wandb.name or cfg.experiment_name
    wandb_mode = "offline" if cfg.wandb.offline else "online"

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        tags=cfg.wandb.tags,
        config=vars(cfg),
        mode=wandb_mode,
    )

    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer: {cfg.data.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name_or_path)

    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    # Build model
    print("Building model...")
    model = LHTEncoder(cfg)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model parameters: {num_params / 1e6:.1f}M ({num_trainable / 1e6:.1f}M trainable)"
    )

    # Watch model with wandb for gradient/weight tracking
    if cfg.wandb.watch_model != "none":
        wandb.watch(
            model,
            log=cfg.wandb.watch_model,
            log_freq=cfg.wandb.watch_freq,
            log_graph=True,
        )

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Setup scheduler
    num_training_steps = cfg.training.num_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Setup mixed precision
    use_amp = cfg.training.mixed_precision in ["fp16", "bf16"]
    scaler = (
        GradScaler() if use_amp and cfg.training.mixed_precision == "fp16" else None
    )
    amp_dtype = (
        torch.bfloat16
        if cfg.training.mixed_precision == "bf16"
        else torch.float16 if use_amp else torch.float32
    )

    # Load dataset
    print("Loading dataset...")
    train_dataset = load_lht_pretrain_dataset(cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # Resume from checkpoint if provided
    start_step = 0
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"]
        print(f"Resumed from step {start_step}")

    # Training setup
    model.train()
    global_step = start_step
    epoch = 0
    best_loss = float("inf")
    accumulated_loss = 0.0
    accumulated_mlm_loss = 0.0
    accumulated_router_loss = 0.0
    tokens_processed = 0
    time_start = time.time()

    print(f"\n{'='*60}")
    print(f"Starting training from step {start_step} to {num_training_steps}")
    print(
        f"Global batch size: {cfg.training.batch_size * cfg.training.grad_accum_steps}"
    )
    print(f"Gradient accumulation steps: {cfg.training.grad_accum_steps}")
    print(f"{'='*60}\n")

    try:
        while global_step < num_training_steps:
            epoch += 1
            epoch_iterator = tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                disable=False,
            )

            for batch_idx, batch in enumerate(epoch_iterator):
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Mask tokens for MLM
                masked_input_ids, labels = mask_tokens_for_mlm(
                    input_ids,
                    tokenizer,
                    mlm_probability=cfg.training.mlm_probability,
                )

                # Forward pass with mixed precision
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    outputs = model(masked_input_ids, attention_mask=attention_mask)

                    # Compute losses
                    mlm_loss = compute_mlm_loss(outputs["mlm_logits"], labels)
                    router_ratio_loss = outputs["router_ratio_loss"]
                    loss = mlm_loss + router_ratio_loss

                    # Scale loss for gradient accumulation
                    loss = loss / cfg.training.grad_accum_steps

                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Accumulate metrics
                accumulated_loss += loss.item() * cfg.training.grad_accum_steps
                accumulated_mlm_loss += mlm_loss.item()
                accumulated_router_loss += router_ratio_loss.item()
                tokens_processed += attention_mask.sum().item()

                # Gradient accumulation step
                if (batch_idx + 1) % cfg.training.grad_accum_steps == 0:
                    # Clip gradients
                    if scaler is not None:
                        scaler.unscale_(optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.training.max_grad_norm
                    )

                    # Optimizer step
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1

                    # Logging
                    if global_step % cfg.training.log_every == 0:
                        # Compute average losses
                        avg_loss = accumulated_loss / cfg.training.log_every
                        avg_mlm_loss = accumulated_mlm_loss / cfg.training.log_every
                        avg_router_loss = (
                            accumulated_router_loss / cfg.training.log_every
                        )

                        # Compute throughput
                        time_elapsed = time.time() - time_start
                        tokens_per_sec = tokens_processed / time_elapsed
                        samples_per_sec = (
                            cfg.training.batch_size
                            * cfg.training.grad_accum_steps
                            * cfg.training.log_every
                            / time_elapsed
                        )

                        # Get learning rate
                        current_lr = scheduler.get_last_lr()[0]

                        # Basic metrics
                        log_dict = {
                            "train/loss": avg_loss,
                            "train/mlm_loss": avg_mlm_loss,
                            "train/router_ratio_loss": avg_router_loss,
                            "train/learning_rate": current_lr,
                            "train/grad_norm": grad_norm.item(),
                            "train/throughput_tokens_per_sec": tokens_per_sec,
                            "train/throughput_samples_per_sec": samples_per_sec,
                            "train/step": global_step,
                            "train/epoch": epoch,
                        }

                        # Router statistics
                        router_stats = log_router_statistics(
                            outputs["hierarchy"], attention_mask, cfg
                        )
                        log_dict.update(router_stats)

                        # Log to wandb
                        wandb.log(log_dict, step=global_step)

                        # Update progress bar
                        epoch_iterator.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "mlm": f"{avg_mlm_loss:.4f}",
                                "lr": f"{current_lr:.2e}",
                            }
                        )

                        # Reset accumulators
                        accumulated_loss = 0.0
                        accumulated_mlm_loss = 0.0
                        accumulated_router_loss = 0.0
                        tokens_processed = 0
                        time_start = time.time()

                    # Evaluation and visualization
                    if global_step % cfg.training.eval_every == 0:
                        print(
                            f"\n[Step {global_step}] Running evaluation and visualization..."
                        )

                        with torch.no_grad():
                            # Sample predictions table
                            try:
                                pred_table = log_sample_predictions(
                                    input_ids,
                                    masked_input_ids,
                                    labels,
                                    outputs["mlm_logits"],
                                    tokenizer,
                                    num_samples=3,
                                    top_k=5,
                                )
                                wandb.log(
                                    {"predictions/samples": pred_table},
                                    step=global_step,
                                )
                            except Exception as e:
                                print(f"Warning: Failed to log sample predictions: {e}")

                            # Gradient flow visualization
                            try:
                                grad_flow_fig = log_gradient_flow(
                                    list(model.named_parameters())
                                )
                                wandb.log(
                                    {"gradients/flow": wandb.Image(grad_flow_fig)},
                                    step=global_step,
                                )
                                plt.close(grad_flow_fig)
                            except Exception as e:
                                print(f"Warning: Failed to log gradient flow: {e}")

                        print(f"[Step {global_step}] Evaluation complete\n")

                    # Checkpointing
                    if global_step % cfg.training.save_every == 0:
                        checkpoint_dir = os.path.join(
                            "checkpoints", cfg.experiment_name
                        )
                        is_best = (
                            avg_loss < best_loss if "avg_loss" in locals() else False
                        )
                        if is_best:
                            best_loss = avg_loss

                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            cfg,
                            checkpoint_dir,
                            is_best=is_best,
                        )

                    # Check if we've reached max steps
                    if global_step >= num_training_steps:
                        break

            if global_step >= num_training_steps:
                break

        print(f"\n{'='*60}")
        print(f"Training complete! Reached {global_step} steps")
        print(f"{'='*60}\n")

        # Save final checkpoint
        checkpoint_dir = os.path.join("checkpoints", cfg.experiment_name)
        save_checkpoint(
            model, optimizer, scheduler, global_step, cfg, checkpoint_dir, is_best=False
        )

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving checkpoint...")
        checkpoint_dir = os.path.join("checkpoints", cfg.experiment_name)
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            global_step,
            cfg,
            checkpoint_dir,
            is_best=False,
        )

    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Finish wandb run
        wandb.finish()


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train LHT with MLM pretraining")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_base.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    train_lht_pretrain(args.config, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
