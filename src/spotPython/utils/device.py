import torch


def getDevice():
    """Get cpu, gpu or mps device for training.
    Returns:
        device (str): Device for training.
        Example:
            >>> from spotPython.utils.device import getDevice
            >>> getDevice()
            'cuda:0'
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device
