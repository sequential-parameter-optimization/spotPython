import numpy as np


def conversion_pred(X) -> np.ndarray:
    """
    Compute conversion predictions for each row in the input array.

    Args:
        X (np.ndarray): 2D array where each row is a configuration.

    Returns:
        np.ndarray: 1D array of conversion predictions.

    Examples:
        >>> import numpy as np
        >>> from spotpython.mo.functions import conversion_pred
        >>> # Example input data
        >>> X = np.array([[1, 2, 3], [4, 5, 6]])
        >>> conversion_pred(X)
        array([  3.5,  19.5])

    """
    return (
        81.09
        + 1.0284 * X[:, 0]
        + 4.043 * X[:, 1]
        + 6.2037 * X[:, 2]
        - 1.8366 * X[:, 0] ** 2
        + 2.9382 * X[:, 1] ** 2
        - 5.1915 * X[:, 2] ** 2
        + 2.2150 * X[:, 0] * X[:, 1]
        + 11.375 * X[:, 0] * X[:, 2]
        - 3.875 * X[:, 1] * X[:, 2]
    )


def activity_pred(X) -> np.ndarray:
    """
    Compute activity predictions for each row in the input array.

    Args:
        X (np.ndarray): 2D array where each row is a configuration.

    Returns:
        np.ndarray: 1D array of activity predictions.

    Examples:
        >>> import numpy as np
        >>> from spotpython.mo.functions import activity_pred
        >>> # Example input data
        >>> X = np.array([[1, 2, 3], [4, 5, 6]])
        >>> activity_pred(X)
        array([  1.5,  10.5])
    """
    return (
        59.85
        + 3.583 * X[:, 0]
        + 0.2546 * X[:, 1]
        + 2.2298 * X[:, 2]
        + 0.83479 * X[:, 0] ** 2
        + 0.07484 * X[:, 1] ** 2
        + 0.05716 * X[:, 2] ** 2
        - 0.3875 * X[:, 0] * X[:, 1]
        - 0.375 * X[:, 0] * X[:, 2]
        + 0.3125 * X[:, 1] * X[:, 2]
    )


def fun_myer16a(X, fun_control=None) -> np.ndarray:
    """
    Compute both conversion and activity predictions for each row in the input array.

    Notes:
        Implements a response surface experiment described by Myers, Montgomery, and Anderson-Cook (2016). The function computes two objectives: conversion and activity.

    References:
        - Myers, R. H., Montgomery, D. C., and Anderson-Cook, C. M. Response surface methodology: process and product optimization using designed experiments. John Wiley & Sons, 2016.
        - Kuhn, M. desirability: Function optimization and ranking via desirability functions. Tech. rep., 9 2016.

    Args:
        X (np.ndarray): 2D array where each row is a configuration.
        fun_control (dict, optional): Additional control parameters (not used here).

    Returns:
        np.ndarray: 2D array where each row contains [conversion_pred, activity_pred].

    Examples:
        >>> import numpy as np
        >>> from spotpython.mo.functions import fun_myer16a
        >>> # Example input data
        >>> X = np.array([[1, 2, 3], [4, 5, 6]])
        >>> fun_myer16a(X)
        array([[  3.5,   1.5],
               [ 19.5,  10.5]])
    """
    return np.column_stack((conversion_pred(X), activity_pred(X)))
