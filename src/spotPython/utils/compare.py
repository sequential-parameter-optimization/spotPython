import numpy as np


def selectNew(A, X, tolerance=0):
    """
    Select rows from A that are not in X.

    Args:
        A (numpy.ndarray): A array with new values
        X (numpy.ndarray): X array with known values

    Returns:
        (numpy.ndarray): array with unknown (new) values
        (numpy.ndarray): array with `True` if value is new, otherwise `False`.
    """
    ind = np.zeros(A.shape[0], dtype=bool)
    for i in range(X.shape[0]):
        B = np.abs(A - X[i, :])
        ind = ind + np.all(B <= tolerance, axis=1)
    return A[~ind], ~ind


def find_equal_in_lists(a, b):
    """Find equal values in two lists.

    Args:
        a (list): list with a values
        b (list): list with b values

    Returns:
        list: list with 1 if equal, otherwise 0
    Example:
        >>> a = [1, 2, 3, 4, 5]
        >>> b = [1, 2, 3, 4, 5]
        >>> find_equal_in_lists(a, b)
        [1, 1, 1, 1, 1]
    """
    equal = []
    for i in range(len(a)):
        if a[i] == b[i]:
            equal.append(1)
        else:
            equal.append(0)
    return equal
