from typing import List, Tuple
import numpy as np
import warnings


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


def remove_nan(X: np.ndarray, y: np.ndarray, stop_on_zero_return: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Remove rows from X and y where y contains NaN values and issue a warning
        if the dimension of the returned y array is smaller than the dimension of the original y array.
        Handles both 1D (shape `(n,)`) and 2D (shape `(n, m)`) y arrays.

    Args:
        X (numpy.ndarray):
            X array
        y (numpy.ndarray):
            y array (can be 1D or 2D)
        stop_on_zero_return (bool):
            whether to stop if the returned dimension is less than 1.
            Default is False.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
            X and y arrays with rows containing NaN values in y removed.

    Examples:
        >>> import numpy as np
        >>> from spotpython.utils.repair import remove_nan
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([1, np.nan, 2])
        >>> X_cleaned, y_cleaned = remove_nan(X, y)
        >>> print(X_cleaned, y_cleaned)
        [[1 2]
         [5 6]] [1. 2.]

        >>> y = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
        >>> X_cleaned, y_cleaned = remove_nan(X, y)
        >>> print(X_cleaned, y_cleaned)
        [[1 2]] [[1. 2.]]
    """
    # Get the original dimension of the y array
    original_dim = y.shape[0]

    # Identify rows where all elements in y are finite
    if y.ndim == 1:
        ind = np.isfinite(y)
    elif y.ndim == 2:
        ind = np.all(np.isfinite(y), axis=0)
    else:
        raise ValueError("y must be a 1D or 2D array.")

    # Update X and y by removing rows with NaN in y
    X_cleaned = X[ind, :]
    y_cleaned = y[ind, :] if y.ndim == 2 else y[ind]

    # Check if dimensions have been reduced
    returned_dim = y_cleaned.shape[0]
    if returned_dim < original_dim:
        warnings.warn(f"\n!!! The dimension of the returned y array is {y_cleaned.shape[0]}, " f"which is smaller than the original dimension {original_dim}.")
        warnings.warn("\n!!! Check whether to continue with the reduced dimension is useful.")

    # Throw an error if the returned dimension is smaller than one
    if returned_dim < 1 and stop_on_zero_return:
        raise ValueError("!!!! The dimension of the returned y array is less than 1. Check the input data.")

    return X_cleaned, y_cleaned


def apply_penalty_NA(y: np.ndarray, penalty_NA: float, sd=0.1, stop_on_zero_return: bool = False, verbosity=0) -> np.ndarray:
    """
    Replaces NaN values in y with a penalty value of penalty_NA and issues a warning if necessary.

    Args:
        y (numpy.ndarray): y array
        penalty_NA (float): penalty value to replace NaN values in y
        sd (float): standard deviation for the random noise added to penalty_NA. Default is 0.1.
        stop_on_zero_return (bool): whether to stop if the returned dimension is less than 1. Default is False.
        verbosity (int): verbosity level. Default is 0.

    Returns:
        numpy.ndarray: y array with NaN values replaced by penalty value

    Examples:
        >>> import numpy as np
        >>> from spotpython.utils.repair import apply_penalty_NA
        >>> y = np.array([1, np.nan, 2])
        >>> y_cleaned = apply_penalty_NA(y, 0)
        >>> print(y_cleaned)
        [1. 0. 2.]
    """
    if not isinstance(y, np.ndarray):
        raise TypeError("Input y must be a numpy array.")

    if not isinstance(penalty_NA, (int, float)):
        return y

    if not isinstance(sd, (int, float)):
        raise TypeError("sd must be a numeric value.")

    if not isinstance(stop_on_zero_return, bool):
        raise TypeError("stop_on_zero_return must be a boolean value.")

    original_dim = y.shape[0]
    nan_ind = ~np.isfinite(y)
    nan_dim = np.sum(nan_ind)

    random_values = np.random.normal(0, sd, y.shape)
    penalty_values = penalty_NA + random_values

    y_cleaned = y.copy()
    y_cleaned[nan_ind] = penalty_values[nan_ind]

    if nan_dim > 1:
        warnings.warn(f"\n!!! The dimension of the returned y array is {y_cleaned.shape[0]}, " f"which is smaller than the original dimension {original_dim}.")
        warnings.warn("\n!!! Check whether continuing with the reduced dimension is useful.")
        if verbosity > 0:
            print(f"y before penalty: {y}. y after penalty: {y_cleaned}")

    if (original_dim - nan_dim) < 1 and stop_on_zero_return:
        raise ValueError("!!!! The dimension of the returned y array is less than 1. Check the input data.")
    return y_cleaned
