import subprocess


def start_tensorboard() -> subprocess.Popen:
    """Starts a tensorboard server in the background.

    Returns:
        process: The process of the tensorboard server.

    Examples:
        >>> process = start_tensorboard()

    """
    cmd = ["tensorboard", "--logdir=./runs"]
    process = subprocess.Popen(cmd)
    return process


def stop_tensorboard(process) -> None:
    """Stops a tensorboard server.

    Args:
        process (subprocess.Popen):
            The process of the tensorboard server.

    Returns:
        None

    Examples:
        >>> process = start_tensorboard()
        >>> stop_tensorboard(process)
    """
    process.terminate()
