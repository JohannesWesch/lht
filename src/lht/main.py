"""
CLI entrypoints wired from pyproject's [project.scripts].

Usage:
  lht-train-pretrain --config configs/pretrain_base.yaml
  lht-train-pretrain --config configs/pretrain_base.yaml --resume-from checkpoints/lht/checkpoint_step_1000.pt
  lht-train-pretrain --config configs/pretrain_base.yaml --wandb-offline
"""

import sys
from pathlib import Path

import click

# Import the full training function
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.train_pretrain import train_lht_pretrain

from .config import load_config


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    default="configs/pretrain_base.yaml",
    help="Path to YAML config file",
)
@click.option(
    "--resume-from",
    type=str,
    default=None,
    help="Path to checkpoint to resume from",
)
@click.option(
    "--wandb-offline",
    is_flag=True,
    help="Run wandb in offline mode",
)
def cli_train_pretrain(
    config: str, resume_from: str = None, wandb_offline: bool = False
):
    """
    Train LHT with masked language modeling and comprehensive W&B logging.

    Example:
        lht-train-pretrain --config configs/pretrain_base.yaml
        lht-train-pretrain --config configs/pretrain_base.yaml --resume-from checkpoints/lht/checkpoint_step_1000.pt
        lht-train-pretrain --config configs/pretrain_base.yaml --wandb-offline
    """
    # Override wandb offline mode if flag is set
    if wandb_offline:
        cfg = load_config(config)
        cfg.wandb.offline = True
        # Save modified config temporarily
        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(vars(cfg), f)
            temp_config_path = f.name

        train_lht_pretrain(temp_config_path, resume_from=resume_from)
    else:
        train_lht_pretrain(config, resume_from=resume_from)


@click.command()
@click.option("--config", "-c", type=str, required=True)
def cli_eval(config: str):
    _cfg = load_config(config)
    print("Eval stub; implement evaluation logic here.")
