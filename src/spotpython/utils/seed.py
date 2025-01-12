import numpy as np
import random
import torch
import os


def set_all_seeds(seed: int):
    """Set the seed for all relevant random number generators to ensure reproducibility.
    This function sets the seed for Python's built-in `random` module, NumPy,
    and PyTorch's CPU and GPU (CUDA) random number generators. It also configures
    PyTorch's settings to improve the reproducibility of experiments, which is
    crucial when debugging or comparing model performances.

    Args:
        seed (int): The seed value to be set for all random number generators.

    Example:
        >>> from spotpython.utils.seed import set_all_seeds
        >>> set_all_seeds(42)
        >>> # Proceed with model initialization or data processing to ensure results can be reproduced
        >>> model = SomeModel()  # Replace with actual model
        >>> train_model(model)   # Replace with actual training function

    Notes:
        - Setting `torch.backends.cudnn.deterministic` to `True` can make computations
          more reproducible but at the potential cost of performance.
        - Additional considerations may be necessary for complete reproducibility
          in distributed or multi-threaded setups.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # Improvements for reproducibility
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)  # Ensuring hash-based functions use a consistent seed
