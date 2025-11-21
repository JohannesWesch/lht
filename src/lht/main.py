"""
CLI entrypoints wired from pyproject's [project.scripts].

Usage:
  lht-train-pretrain --config configs/pretrain_base.yaml
"""

import click

from .config import load_config
from .dataset import load_lht_pretrain_dataset
from .model import LHTEncoder
from .utils import set_seed


@click.command()
@click.option("--config", "-c", type=str, required=True)
def cli_train_pretrain(config: str):
    cfg = load_config(config)
    set_seed(cfg.seed)

    _dataset, _tokenizer = load_lht_pretrain_dataset(cfg)
    _model = LHTEncoder(cfg.model).to(cfg.device)

    # TODO: implement DataLoader, optimizer, training loop.
    print("Loaded config and model; ready to implement training loop.")


@click.command()
@click.option("--config", "-c", type=str, required=True)
def cli_eval(config: str):
    _cfg = load_config(config)
    print("Eval stub; implement evaluation logic here.")
