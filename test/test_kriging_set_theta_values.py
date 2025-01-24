import pytest
import numpy as np
from spotpython.build.kriging import Kriging

def test_set_theta_values_zero_init():
    # Create a Kriging instance with theta_init_zero=True
    S = Kriging(seed=124, n_theta=2, n_p=2, optim_p=True, noise=True, theta_init_zero=True)

    # Initialize variables
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    S._initialize_variables(nat_X, nat_y)
    S._set_variable_types()

    # Set theta values
    S._set_theta_values()

    # Check if theta values are initialized to zeros
    np.testing.assert_array_equal(S.theta, np.zeros(2, dtype=float))

def test_set_theta_values_non_zero_init():
    # Create a Kriging instance with theta_init_zero=False
    S = Kriging(seed=124, n_theta=2, n_p=2, optim_p=True, noise=True, theta_init_zero=False)

    # Initialize variables
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    S._initialize_variables(nat_X, nat_y)
    S._set_variable_types()

    # Set theta values
    S._set_theta_values()

    # Check if theta values are initialized based on n and k
    expected_theta = np.ones(2, dtype=float) * S.n / (100 * S.k)
    np.testing.assert_array_equal(S.theta, expected_theta)

def test_set_theta_values_adjust_n_theta():
    # Create a Kriging instance with n_theta greater than k
    S = Kriging(seed=124, n_theta=3, n_p=2, optim_p=True, noise=True, theta_init_zero=True)

    # Initialize variables
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    S._initialize_variables(nat_X, nat_y)
    S._set_variable_types()

    # Set theta values
    S._set_theta_values()

    # Check if n_theta is adjusted to k and theta values are initialized to zeros
    assert S.n_theta == S.k
    np.testing.assert_array_equal(S.theta, np.zeros(S.k, dtype=float))

if __name__ == "__main__":
    pytest.main()