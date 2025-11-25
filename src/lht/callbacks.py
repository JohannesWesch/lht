"""
PyTorch Lightning callbacks for visualization and monitoring.

Includes callbacks for:
- Attention pattern visualization
- MLM prediction logging
- Gradient flow monitoring
"""

import torch
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger

from lht.visualization import (
    compute_attention_entropy,
    create_attention_heatmap,
    log_gradient_flow,
    log_sample_predictions,
)


class VisualizationCallback(Callback):
    """
    Callback to log visualizations to W&B during training.

    Logs:
    - Attention heatmaps (every N steps)
    - MLM predictions table (every N steps)
    - Gradient flow (every N steps)
    - Attention entropy statistics (every step)
    """

    def __init__(
        self,
        log_every_n_steps: int = 1000,
        num_samples: int = 3,
        max_tokens_heatmap: int = 60,
    ):
        """
        Args:
            log_every_n_steps: How often to log visualizations
            num_samples: Number of samples to show in prediction table
            max_tokens_heatmap: Maximum tokens to show in attention heatmap
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.num_samples = num_samples
        self.max_tokens_heatmap = max_tokens_heatmap
        self._original_input_ids = None
        self._masked_input_ids = None
        self._labels = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Store original input_ids before masking for visualization."""
        # Store a copy of the batch for visualization
        if batch_idx % self.log_every_n_steps == 0:
            # Get the original input_ids (before masking)
            # Note: This assumes your data collator stores original_input_ids
            self._original_input_ids = batch.get("original_input_ids", None)
            self._masked_input_ids = batch["input_ids"].clone()
            self._labels = batch["labels"].clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log visualizations after training step."""
        if batch_idx % self.log_every_n_steps != 0:
            return

        # Only log if we have a W&B logger
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is None:
            return

        with torch.no_grad():
            # Get tokenizer from data collator
            tokenizer = trainer.datamodule.train_dataloader().collate_fn.tokenizer

            # 1. Log MLM predictions
            if self._original_input_ids is not None:
                # Get model predictions for visualization
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", None)
                positions = batch.get("positions", None)

                model_outputs = pl_module(
                    input_ids, attention_mask=attention_mask, positions=positions
                )
                logits = model_outputs["mlm_logits"]

                prediction_table = log_sample_predictions(
                    input_ids=self._original_input_ids[: self.num_samples],
                    masked_input_ids=self._masked_input_ids[: self.num_samples],
                    labels=self._labels[: self.num_samples],
                    logits=logits[: self.num_samples],
                    tokenizer=tokenizer,
                    num_samples=self.num_samples,
                    top_k=5,
                )
                wandb_logger.log_table(
                    key=f"predictions/step_{trainer.global_step}",
                    dataframe=prediction_table,
                )

            # 2. Log attention heatmap (if model returns attention weights)
            # Note: You'll need to modify your model to optionally return attention weights
            # For now, we'll skip this unless attention weights are available
            if (
                hasattr(model_outputs, "attentions")
                and model_outputs.attentions is not None
            ):
                # Take first sample, first layer attention
                attn_weights = model_outputs.attentions[0][0]  # [H, N, N]

                # Decode tokens for labels
                tokens = [
                    tokenizer.decode([tid])
                    for tid in input_ids[0][: self.max_tokens_heatmap]
                ]

                # Create heatmap
                fig = create_attention_heatmap(
                    attention_weights=attn_weights[
                        :, : self.max_tokens_heatmap, : self.max_tokens_heatmap
                    ],
                    tokens=tokens,
                    layer_name=f"Layer 0 - Step {trainer.global_step}",
                    max_tokens=self.max_tokens_heatmap,
                )

                wandb_logger.log_image(
                    key="attention/layer_0", images=[fig], step=trainer.global_step
                )

                # Compute and log entropy
                entropy = compute_attention_entropy(attn_weights.unsqueeze(0))
                wandb_logger.experiment.log(
                    {
                        "attention/entropy_mean": entropy.mean().item(),
                        "attention/entropy_std": entropy.std().item(),
                        "attention/entropy_min": entropy.min().item(),
                        "attention/entropy_max": entropy.max().item(),
                        "global_step": trainer.global_step,
                    }
                )

            # 3. Log gradient flow
            fig = log_gradient_flow(list(pl_module.named_parameters()))
            wandb_logger.log_image(
                key="gradients/flow", images=[fig], step=trainer.global_step
            )

    def _get_wandb_logger(self, trainer):
        """Get W&B logger if available."""
        if trainer.logger is None:
            return None

        if isinstance(trainer.logger, WandbLogger):
            return trainer.logger

        # Check if it's a list of loggers
        if hasattr(trainer, "loggers"):
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    return logger

        return None


class AttentionEntropyMonitor(Callback):
    """
    Lightweight callback to monitor attention entropy without heavy visualizations.

    Logs entropy statistics every step.
    """

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log attention entropy statistics."""
        # Only if model returns attention weights
        if not hasattr(outputs, "attentions") or outputs.attentions is None:
            return

        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is None:
            return

        with torch.no_grad():
            # Compute entropy for all layers
            for layer_idx, attn in enumerate(outputs.attentions):
                entropy = compute_attention_entropy(attn)

                wandb_logger.experiment.log(
                    {
                        f"attention_entropy/layer_{layer_idx}_mean": entropy.mean().item(),
                        f"attention_entropy/layer_{layer_idx}_std": entropy.std().item(),
                        "global_step": trainer.global_step,
                    }
                )

    def _get_wandb_logger(self, trainer):
        """Get W&B logger if available."""
        if trainer.logger is None:
            return None

        if isinstance(trainer.logger, WandbLogger):
            return trainer.logger

        if hasattr(trainer, "loggers"):
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    return logger

        return None
