"""
Miscellaneous utilities: seeding, logging, checkpointing, etc.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
