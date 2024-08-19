from typing import Tuple
from typing import List

import numpy as np


def selectNew(A: np.ndarray, X: np.ndarray, tolerance: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select rows from A that are not in X.

    Args:
        A (numpy.ndarray): A array with new values
        X (numpy.ndarray): X array with known values
        tolerance (float): tolerance value for comparison

    Returns:
        (numpy.ndarray): array with unknown (new) values
        (numpy.ndarray): array with `True` if value is new, otherwise `False`.

    Examples:
    >>> from spotpython.utils.compare import selectNew
        import numpy as np
        A = np.array([[1,2,3],[4,5,6]])
        X = np.array([[1,2,3],[4,5,6]])
        B, ind  = selectNew(A, X)
        assert B.shape[0] == 0
        assert np.equal(ind, np.array([False, False])).all()
    >>> from spotpython.utils.compare import selectNew
        A = np.array([[1,2,3],[4,5,7]])
        X = np.array([[1,2,3],[4,5,6]])
        B, ind  = selectNew(A, X)
        assert B.shape[0] == 1
        assert np.equal(ind, np.array([False, True])).all()
    """
    B = np.abs(A[:, None] - X)
    ind = np.any(np.all(B <= tolerance, axis=2), axis=1)
    return A[~ind], ~ind


def find_equal_in_lists(a: List[int], b: List[int]) -> List[int]:
    """Find equal values in two lists.

    Args:
        a (list): list with a values
        b (list): list with b values

    Returns:
        list: list with 1 if equal, otherwise 0

    Examples:
        >>> from spotpython.utils.compare import find_equal_in_lists
            a = [1, 2, 3, 4, 5]
            b = [1, 2, 3, 4, 5]
            find_equal_in_lists(a, b)
            [1, 1, 1, 1, 1]
    """
    equal = [1 if a[i] == b[i] else 0 for i in range(len(a))]
    return equal
