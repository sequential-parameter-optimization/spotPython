from numpy import around
from numpy import isfinite


def repair_non_numeric(X, var_type):
    """
    Round non-numeric values to integers.
    This applies to all variables except for "num" and "float".

    Args:
        X (numpy.ndarray): X array
        var_type (list): list with type information
    """
    for i in range(X.shape[1]):
        if var_type[i] not in ["num", "float"]:
            X[:, i] = around(X[:, i])
    return X


def remove_nan(X, y):
    ind = isfinite(y)
    y = y[ind]
    X = X[ind, :]
    return X, y
