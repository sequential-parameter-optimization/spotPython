from sys import stdout


def progress_bar(progress, bar_length=10, message="spotPython tuning:"):
    """
    Displays or updates a console progress bar.
    See: https://stackoverflow.com/a/15860757

    Args:
        (float) progress: a float between 0 and 1. Any int will be converted to a float.
        A value under 0 represents a halt.
        A value at 1 or bigger represents 100%.
        A message text.
    """

    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(bar_length * progress))
    text = message + " [{0}] {1:.2f}% {2}\r".format("#" * block + "-" * (bar_length - block), progress * 100, status)
    stdout.write(text)
    stdout.flush()
