"""
PyTorch Lightning Module for LHT.

This module wraps the LHTEncoder and handles:
- Training step (MLM loss)
- Validation step
- Optimization configuration (AdamW, Scheduler)
"""

import torch.optim as optim
from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup

from lht.config import ExperimentConfig
from lht.model import LHTEncoder
from lht.training import compute_mlm_loss


class LHTLightningModule(LightningModule):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = LHTEncoder(config)

        # Important: set automatic_optimization to True (default) for simple mixed precision handling
        # self.automatic_optimization = True

    def forward(self, input_ids, attention_mask=None, positions=None):
        return self.model(input_ids, attention_mask=attention_mask, positions=positions)

    def training_step(self, batch, batch_idx):
        # Batch from DataCollatorForLanguageModeling is already masked if mlm=True
        # Keys: input_ids, attention_mask, labels
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # Extract positions if present (added by HierarchicalDataCollator)
        positions = batch.get("positions", None)

        # Forward pass
        outputs = self(input_ids, attention_mask=attention_mask, positions=positions)

        # Compute loss
        loss = compute_mlm_loss(outputs["mlm_logits"], labels)

        # Logging with explicit batch size
        batch_size = input_ids.shape[0]
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/lr",
            self.lr_schedulers().get_last_lr()[0],
            on_step=True,
            prog_bar=False,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        positions = batch.get("positions", None)

        outputs = self(input_ids, attention_mask=attention_mask, positions=positions)
        loss = compute_mlm_loss(outputs["mlm_logits"], labels)

        batch_size = input_ids.shape[0]
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def configure_optimizers(self):
        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=self.config.training.num_steps,
        )

        # Lightning requires a specific dictionary format for schedulers that update per step
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # update every step
                "frequency": 1,
            },
        }
