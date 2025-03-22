import numpy as np
import pandas as pd
import pytest
from spotpython.design.utils import get_boundaries, generate_search_grid


def test_get_boundaries_with_positive_numbers():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    min_values, max_values = get_boundaries(data)
    assert np.array_equal(min_values, np.array([1, 2, 3]))
    assert np.array_equal(max_values, np.array([7, 8, 9]))


def test_get_boundaries_with_negative_numbers():
    data = np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]])
    min_values, max_values = get_boundaries(data)
    assert np.array_equal(min_values, np.array([-7, -8, -9]))
    assert np.array_equal(max_values, np.array([-1, -2, -3]))


def test_get_boundaries_with_mixed_numbers():
    data = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]])
    min_values, max_values = get_boundaries(data)
    assert np.array_equal(min_values, np.array([-4, -8, -6]))
    assert np.array_equal(max_values, np.array([7, 5, 9]))


def test_get_boundaries_with_single_row():
    data = np.array([[1, 2, 3]])
    min_values, max_values = get_boundaries(data)
    assert np.array_equal(min_values, np.array([1, 2, 3]))
    assert np.array_equal(max_values, np.array([1, 2, 3]))


def test_get_boundaries_with_single_column():
    data = np.array([[1], [4], [7]])
    min_values, max_values = get_boundaries(data)
    assert np.array_equal(min_values, np.array([1]))
    assert np.array_equal(max_values, np.array([7]))


def test_get_boundaries_with_empty_array():
    data = np.array([[]])
    with pytest.raises(ValueError):
        get_boundaries(data)


def test_generate_search_grid_numpy():
    x_min = np.array([0, 0])
    x_max = np.array([1, 1])
    grid = generate_search_grid(x_min, x_max, n_points=3)
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (9, 2)


def test_generate_search_grid_pandas():
    x_min = np.array([0, 0])
    x_max = np.array([1, 1])
    col_names = ["x", "y"]
    grid = generate_search_grid(x_min, x_max, n_points=3, col_names=col_names)
    assert isinstance(grid, pd.DataFrame)
    assert grid.shape == (9, 2)
    assert list(grid.columns) == col_names


def test_generate_search_grid_different_n_points():
    x_min = np.array([0, 0])
    x_max = np.array([1, 1])
    grid = generate_search_grid(x_min, x_max, n_points=5)
    assert grid.shape == (25, 2)


def test_generate_search_grid_different_ranges():
    x_min = np.array([1, 2])
    x_max = np.array([4, 5])
    grid = generate_search_grid(x_min, x_max, n_points=3)
    assert np.allclose(grid[0], np.array([1.0, 2.0]))
    assert np.allclose(grid[-1], np.array([4.0, 5.0]))


def test_generate_search_grid_col_names_mismatch():
    x_min = np.array([0, 0])
    x_max = np.array([1, 1])
    col_names = ["x"]
    with pytest.raises(ValueError):
        generate_search_grid(x_min, x_max, col_names=col_names)