import warnings
import numpy as np
from spotPython.utils.repair import remove_nan
import pytest


def test_remove_nan_no_nan():
    # Case where y has no NaNs
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    X_cleaned, y_cleaned = remove_nan(X, y)
    np.testing.assert_array_equal(X, X_cleaned)
    np.testing.assert_array_equal(y, y_cleaned)


def test_remove_nan_with_nan():
    # Case where y contains NaN and rows are removed
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([np.nan, 2, 3])
    X_cleaned, y_cleaned = remove_nan(X, y)
    np.testing.assert_array_equal(X_cleaned, np.array([[3, 4], [5, 6]]))
    np.testing.assert_array_equal(y_cleaned, np.array([2, 3]))


def test_remove_nan_dimension_warning():
    # Case that should trigger a warning
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, np.nan, np.nan])
    with pytest.warns(UserWarning):
        X_cleaned, y_cleaned = remove_nan(X, y)
    assert y_cleaned.shape[0] < y.shape[0], "Expected dimension reduction did not trigger a warning."


def test_remove_nan_dimension_error():
    # Case that should raise ValueError
    X = np.array([[1, 2]])
    y = np.array([np.nan])
    with pytest.raises(ValueError):
        X_cleaned, y_cleaned = remove_nan(X, y)
