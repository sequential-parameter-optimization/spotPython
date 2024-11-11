from sys import stdout


def progress_bar(progress: float, bar_length: int = 10, message: str = "spotpython tuning:", y=None, filename=None) -> None:
    """
    Displays or updates a console progress bar.

    Args:
        progress (float):
            a float between 0 and 1. Any int will be converted to a float.
            A value under 0 represents a halt.
            A value at 1 or bigger represents 100%.
        bar_length (int):
            length of the progress bar
        message (str):
            message text to display
        filename (str):
            If not None, write the progress bar to filename.
    """
    if filename is not None:
        # open the file in append mode
        file = open(filename, "a")
    status = ""
    if y is not None:
        message = f"{message} {y}"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    elif progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(bar_length * progress))
    text = f"{message} [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}% {status}\r\n"
    if filename is not None:
        file.write(text)
        file.flush()
    stdout.write(text)
    stdout.flush()
    if filename is not None:
        file.close()
