import torch


def getDevice(device=None):
    """Get CPU, GPU (CUDA), or MPS device for training.

    Args:
        device (str):
            Device for training. If None or "auto", the device is selected automatically based on availability.

    Returns:
        device (str):
            Device for training.

    Raises:
        ValueError: If the requested device is not recognized or available.

    Examples:
        >>> from spotpython.utils.device import getDevice
            getDevice()
                'cuda:0'
    """
    if device is None or device == "auto":
        # Automatically select device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        return device

    # Check the explicit device request
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but no CUDA device is available.")
    elif device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS device requested but MPS is not available.")
    elif device == "cpu":
        return "cpu"
    else:
        raise ValueError(f"Unrecognized device: {device}. Valid options are 'cpu', 'cuda:x', or 'mps'.")

    return device
