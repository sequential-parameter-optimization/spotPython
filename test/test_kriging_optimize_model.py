import pytest
import numpy as np
from spotpython.build import Kriging

def test_optimize_model_example_1():
    # Example 1 from the docstring
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    n = 2
    p = 2
    S = Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
    S._initialize_variables(nat_X, nat_y)
    S._set_variable_types()
    S._set_theta_values()
    S._initialize_matrices()
    S._set_de_bounds()
    new_theta_p_Lambda = S._optimize_model()
    assert len(new_theta_p_Lambda) == n + p + 1

def test_optimize_model_example_2():
    # Example 2 from the docstring
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    n_theta = 2
    n_p = 2
    S = Kriging(seed=124, n_theta=n_theta, n_p=n_p, optim_p=True, noise=True)
    S._initialize_variables(nat_X, nat_y)
    S._set_variable_types()
    S._set_theta_values()
    S._initialize_matrices()
    S._set_de_bounds()
    new_theta_p_Lambda = S._optimize_model()
    assert len(new_theta_p_Lambda) == n_theta + n_p + 1

def test_optimize_model_example_3():
    # Example 3 from the docstring
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    n_theta = 2
    n_p = 2
    S = Kriging(seed=124, n_theta=n_theta, n_p=n_p, optim_p=True, noise=False)
    S._initialize_variables(nat_X, nat_y)
    S._set_variable_types()
    S._set_theta_values()
    S._initialize_matrices()
    S._set_de_bounds()
    new_theta_p_Lambda = S._optimize_model()
    assert len(new_theta_p_Lambda) == n_theta + n_p

def test_optimize_model_example_4():
    # Example 4 from the docstring
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    n_theta = 2
    n_p = 1
    S = Kriging(seed=124, n_theta=n_theta, n_p=n_p, optim_p=True, noise=False)
    S._initialize_variables(nat_X, nat_y)
    S._set_variable_types()
    S._set_theta_values()
    S._initialize_matrices()
    S._set_de_bounds()
    new_theta_p_Lambda = S._optimize_model()
    assert len(new_theta_p_Lambda) == n_theta + n_p

def test_optimize_model_example_5():
    # Example 5 from the docstring
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    n_theta = 1
    n_p = 1
    S = Kriging(seed=124, n_theta=n_theta, n_p=n_p, optim_p=False, noise=False)
    S._initialize_variables(nat_X, nat_y)
    S._set_variable_types()
    S._set_theta_values()
    S._initialize_matrices()
    S._set_de_bounds()
    new_theta_p_Lambda = S._optimize_model()
    assert len(new_theta_p_Lambda) == 1

if __name__ == "__main__":
    pytest.main()