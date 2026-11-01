"""Deterministic seeding for reproducibility."""
import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """Set deterministic seeds for Python, NumPy, PyTorch, and CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    logger.info(f"All seeds set to {seed}, deterministic={deterministic}")

