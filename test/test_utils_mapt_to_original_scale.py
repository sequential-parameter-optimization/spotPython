import pytest
import numpy as np
import pandas as pd
from spotpython.design.utils import map_to_original_scale

def test_map_to_original_scale_with_dataframe():
    X_search = pd.DataFrame([[0.5, 0.5], [0.25, 0.75]], columns=['x', 'y'])
    x_min = np.array([0, 0])
    x_max = np.array([10, 20])
    expected = pd.DataFrame([[5.0, 10.0], [2.5, 15.0]], columns=['x', 'y'])

    result = map_to_original_scale(X_search, x_min, x_max)

    pd.testing.assert_frame_equal(result, expected)

def test_map_to_original_scale_with_numpy_array():
    X_search = np.array([[0.5, 0.5], [0.25, 0.75]])
    x_min = np.array([0, 0])
    x_max = np.array([10, 20])
    expected = np.array([[5.0, 10.0], [2.5, 15.0]])

    result = map_to_original_scale(X_search, x_min, x_max)

    np.testing.assert_array_almost_equal(result, expected)

def test_map_to_original_scale_invalid_input_type():
    X_search = [[0.5, 0.5], [0.25, 0.75]]  # Not a DataFrame or NumPy array
    x_min = np.array([0, 0])
    x_max = np.array([10, 20])

    with pytest.raises(TypeError, match="X_search must be a Pandas DataFrame or a NumPy array."):
        map_to_original_scale(X_search, x_min, x_max)

def test_map_to_original_scale_mismatched_dimensions():
    X_search = np.array([[0.5, 0.5], [0.25, 0.75]])
    x_min = np.array([0])
    x_max = np.array([10])

    with pytest.raises(IndexError, match="x_min and X_search must have the same number of columns."):
        map_to_original_scale(X_search, x_min, x_max)