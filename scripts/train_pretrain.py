"""
Comprehensive pretraining script for LHT with PyTorch Lightning.

This script implements the full training loop using PyTorch Lightning Trainer:
- Masked Language Modeling objective
- W&B logging
- Mixed precision training
- Gradient accumulation
- Checkpointing
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lht.config import load_config
from lht.data.mlm import HierarchicalMLMDataModule, MLMDataModule
from lht.lightning_module import LHTLightningModule
from lht.utils import set_seed


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train LHT with MLM pretraining (PyTorch Lightning)"
    )
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
    config_path = args.config
    resume_from = args.resume_from

    # Load config
    print(f"Loading config from {config_path}...")
    cfg = load_config(config_path)

    # Set seed for reproducibility
    set_seed(cfg.seed)
    pl.seed_everything(cfg.seed)

    # Initialize DataModule
    print("Initializing Data Module...")
    if cfg.model.geometry:  # Check if using hierarchical model
        data_module = HierarchicalMLMDataModule(cfg)
    else:
        data_module = MLMDataModule(cfg)

    # Initialize Lightning Module
    print("Initializing Lightning Module...")
    model = LHTLightningModule(cfg)

    # Initialize Loggers
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name or cfg.experiment_name,
        tags=cfg.wandb.tags,
        config=vars(cfg),
        offline=cfg.wandb.offline,
        log_model=cfg.wandb.log_model,
    )

    # Initialize Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.experiment_name}",
            filename="checkpoint-{step}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            save_last=True,
            every_n_train_steps=cfg.training.save_every,
        ),
    ]

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = pl.Trainer(
        max_steps=cfg.training.num_steps,
        accumulate_grad_batches=cfg.training.grad_accum_steps,
        gradient_clip_val=cfg.training.max_grad_norm,
        precision=(
            cfg.training.mixed_precision
            if cfg.training.mixed_precision in ["16-mixed", "bf16-mixed", "32"]
            else "16-mixed"
        ),  # Map to PL precision strings
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator=cfg.device if cfg.device in ["cuda", "cpu", "mps"] else "auto",
        devices=1,  # Adjust based on your setup or config
        log_every_n_steps=cfg.training.log_every,
        val_check_interval=cfg.training.eval_every,
    )

    # Start Training
    print("Starting training...")
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_from)


if __name__ == "__main__":
    main()
