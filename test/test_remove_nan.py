import numpy as np
from spotpython.utils.repair import remove_nan
import pytest
import warnings


def test_remove_nan_dimension_error():
    # Case that should raise ValueError
    X = np.array([[1, 2]])
    y = np.array([np.nan])
    with pytest.raises(ValueError):
        X_cleaned, y_cleaned = remove_nan(X, y, stop_on_zero_return=True)


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


def test_remove_nan_basic():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, np.nan, 2])
    X_cleaned, y_cleaned = remove_nan(X, y)
    np.testing.assert_array_equal(X_cleaned, np.array([[1, 2], [5, 6]]))
    np.testing.assert_array_equal(y_cleaned, np.array([1, 2]))


def test_remove_nan_warning():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, np.nan, 2])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X_cleaned, y_cleaned = remove_nan(X, y)
        assert len(w) == 2
        assert issubclass(w[-1].category, UserWarning)
        assert "smaller than the original dimension" in str(w[0].message)
        assert "Check whether to continue with the reduced dimension is useful" in str(w[1].message)


def test_remove_nan_value_error():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([np.nan, np.nan, np.nan])
    with pytest.raises(ValueError):
        remove_nan(X, y, stop_on_zero_return=True)


def test_no_nan():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    X_cleaned, y_cleaned = remove_nan(X, y)
    np.testing.assert_array_equal(X_cleaned, X)
    np.testing.assert_array_equal(y_cleaned, y)


def test_remove_nan_empty_X():
    X = np.array([[], [], []])
    y = np.array([1, np.nan, 2])
    X_cleaned, y_cleaned = remove_nan(X, y)
    np.testing.assert_array_equal(X_cleaned, np.array([[], []]))
    np.testing.assert_array_equal(y_cleaned, np.array([1, 2]))
