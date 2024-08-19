import datetime


def get_timestamp(only_int=True) -> str:
    """
    Returns a timestamp as a string.

    Args:
        only_int (bool): if True, the timestamp is returned as an integer.

    Returns:
        str: the timestamp as a string.

    Examples:
        >>> from spotpython.utils.time import get_timestamp
        >>> get_timestamp()
        '2021-06-28 14:51:54.500000'
        >>> get_timestamp(only_int=True)
        '20210628145154500000'
    """
    dt = datetime.datetime.now().isoformat(sep=" ", timespec="microseconds")
    if only_int:
        # remove - . : and space
        dt = dt.replace("-", "")
        dt = dt.replace(".", "")
        dt = dt.replace(":", "")
        dt = dt.replace(" ", "")
    return dt
