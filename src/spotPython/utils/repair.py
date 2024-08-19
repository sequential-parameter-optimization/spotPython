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
        Issues a ValueError if the dimension of the returned y array is less than 21 and
        stop_on_zero_return is True.

    Args:
        X (numpy.ndarray):
            X array
        y (numpy.ndarray):
            y array
        stop_on_zero_return (bool):
            whether to stop if the returned dimension is less than 1.
            Default is False.

    Returns:
        Tuple[numpy.ndarray, np.ndarray]:
            X and y arrays with rows containing NaN values in y removed.

    Examples:
        >>> import numpy as np
            from spotpython.utils.repair import remove_nan
            X = np.array([[1, 2], [3, 4], [5, 6]])
            y = np.array([1, np.nan, 2])
            X_cleaned, y_cleaned = remove_nan(X, y)
            print(X_cleaned, y_cleaned)
            [[1 2]
             [5 6]] [1. 2.]
    """
    # Get the original dimension of the y array
    original_dim = y.shape[0]

    # Identify indices where y is not NaN
    ind = np.isfinite(y)

    # Update X and y by removing rows with NaN in y
    X_cleaned = X[ind, :]
    y_cleaned = y[ind]

    # Check if dimensions have been reduced
    returned_dim = y_cleaned.shape[0]
    if returned_dim < original_dim:
        warnings.warn(
            f"\n!!! The dimension of the returned y array is {y_cleaned.shape[0]}, "
            f"which is smaller than the original dimension {original_dim}."
        )
        warnings.warn("\n!!! Check whether to continue with the reduced dimension is useful.")
    # throw an error if the returned dimension is smaller than one
    if returned_dim < 1 and stop_on_zero_return:
        raise ValueError("!!!! The dimension of the returned y array is less than 1. Check the input data.")

    return X_cleaned, y_cleaned
