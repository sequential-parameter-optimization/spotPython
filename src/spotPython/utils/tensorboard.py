import subprocess


def start_tensorboard() -> subprocess.Popen:
    """Starts a tensorboard server in the background.

    Returns:
        process: The process of the tensorboard server.

    Examples:
        >>> from spotpython.utils.tensorboard import start_tensorboard
        >>> process = start_tensorboard()

    """
    cmd = ["tensorboard", "--logdir=./runs"]
    process = subprocess.Popen(cmd)
    return process


def stop_tensorboard(process) -> None:
    """
    Stops a tensorboard server if the process exists.

    Args:
        process (subprocess.Popen): The process of the tensorboard server.

    Returns:
        None

    Examples:
        >>> from spotpython.utils.tensorboard import start_tensorboard, stop_tensorboard
        >>> process = start_tensorboard()
        >>> stop_tensorboard(process)
    """
    if process is not None and process.poll() is None:
        process.terminate()
        process.wait()  # Ensure the process has terminated
    else:
        print("No active tensorboard process found or the process is already terminated.")
