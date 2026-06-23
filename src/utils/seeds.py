"""
Centralized random seed management.

Every script and notebook must call `set_all_seeds()` before any
stochastic operation. This is the single most important utility for
reproducibility — see DEVELOPMENT.md §6.1.
"""
from __future__ import annotations

import os
import random

import numpy as np

from src.config import RANDOM_SEED


def set_all_seeds(seed: int = RANDOM_SEED) -> None:
    """Seed all sources of randomness for reproducibility.

    Seeds Python's `random`, NumPy, PyTorch (CPU and CUDA), and configures
    cuDNN for deterministic operation.

    Args:
        seed: The seed value. Defaults to the project-wide `RANDOM_SEED`.

    Note:
        Setting `torch.backends.cudnn.deterministic = True` and
        `torch.backends.cudnn.benchmark = False` reduces GPU throughput
        slightly but is required for bitwise-reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (imported lazily to avoid forcing the dependency at import time)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_seed() -> int:
    """Return the project-wide seed value."""
    return RANDOM_SEED
