from typing import List, Tuple
import numpy as np


def repair_non_numeric(X: np.ndarray, var_type: List[str]) -> np.ndarray:
    """
    Round non-numeric values to integers.
    This applies to all variables except for "num" and "float".

    Args:
        X (numpy.ndarray): X array
        var_type (list): list with type information

    Returns:
        numpy.ndarray: X array with non-numeric values rounded to integers

    Examples:
        >>> X = np.array([[1.2, 2.3], [3.4, 4.5]])
        >>> var_type = ["num", "factor"]
        >>> repair_non_numeric(X, var_type)
        array([[1., 2.],
               [3., 4.]])
    """
    mask = np.isin(var_type, ["num", "float"], invert=True)
    X[:, mask] = np.around(X[:, mask])
    return X


def remove_nan(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove rows from X and y where y contains NaN values.

    Args:
        X (numpy.ndarray): X array
        y (numpy.ndarray): y array

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: X and y arrays with rows containing NaN values in y removed

    Examples:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([1, np.nan, 2])
        >>> remove_nan(X, y)
        (array([[1, 2],
                [5, 6]]), array([1., 2.]))
    """
    ind = np.isfinite(y)
    y = y[ind]
    X = X[ind, :]
    return X, y
