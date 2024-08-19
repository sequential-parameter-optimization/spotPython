import torch


def getDevice(device=None):
    """Get cpu, gpu or mps device for training.
    Args:
        device (str):
            Device for training. If None or "auto" the device is selected automatically.

    Returns:
        device (str):
            Device for training.

    Examples:
        >>> from spotpython.utils.device import getDevice
        >>> getDevice()
        'cuda:0'
    """
    if device is None or device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
    return device
