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
