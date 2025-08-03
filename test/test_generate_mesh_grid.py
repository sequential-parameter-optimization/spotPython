import numpy as np
import pytest
from spotpython.surrogate.plot import generate_mesh_grid

def test_generate_mesh_grid_with_X():
    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    X_i, X_j, grid_points = generate_mesh_grid(X, i=0, j=1, num=3)
    assert X_i.shape == (3, 3)
    assert X_j.shape == (3, 3)
    assert grid_points.shape == (9, 3)
    # Check that the remaining dimension is filled with the mean
    np.testing.assert_allclose(grid_points[:, 2], np.mean(X[:, 2]))

def test_generate_mesh_grid_with_bounds():
    lower = np.array([0, 10, 100])
    upper = np.array([2, 12, 102])
    X_i, X_j, grid_points = generate_mesh_grid(lower=lower, upper=upper, i=0, j=1, num=2)
    assert X_i.shape == (2, 2)
    assert X_j.shape == (2, 2)
    assert grid_points.shape == (4, 3)
    # Check that the remaining dimension is filled with the mean of bounds
    assert np.all(grid_points[:, 2] == 101.0)

def test_generate_mesh_grid_var_type_floor():
    X = np.array([
        [1.2, 2.7, 3.5],
        [4.8, 5.1, 6.9],
        [7.6, 8.3, 9.2]
    ])
    var_type = ["int", "num", "factor"]
    X_i, X_j, grid_points = generate_mesh_grid(X, i=0, j=1, num=3, var_type=var_type, use_floor=True)
    # Check that int/factor columns are floored
    assert np.all(np.equal(grid_points[:, 0], np.floor(grid_points[:, 0] + 0.5)))
    assert np.allclose(grid_points[:, 1], grid_points[:, 1])  # num column unchanged
    assert np.all(np.equal(grid_points[:, 2], np.floor(grid_points[:, 2] + 0.5)))

def test_generate_mesh_grid_invalid_args():
    X = np.random.rand(3, 2)
    lower = np.array([0, 1])
    upper = np.array([2, 3])
    # Both X and bounds provided
    with pytest.raises(ValueError):
        generate_mesh_grid(X, i=0, j=1, lower=lower, upper=upper)
    # Neither X nor bounds provided
    with pytest.raises(ValueError):
        generate_mesh_grid(i=0, j=1)