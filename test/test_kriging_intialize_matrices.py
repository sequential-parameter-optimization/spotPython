import pytest
import numpy as np
from numpy import log, var
from spotpython.build import Kriging

def test_initialize_matrices():
    # Create a Kriging instance
    S = Kriging(seed=124, n_theta=2, n_p=2, optim_p=True, noise=True)

    # Initialize variables
    nat_X = np.array([[1, 2], [3, 4], [5, 6]])
    nat_y = np.array([1, 2, 3])
    S._initialize_variables(nat_X, nat_y)
    S._set_variable_types()
    S._set_theta_values()

    # Initialize matrices
    S._initialize_matrices()

    # Check if the matrices and attributes are initialized correctly
    np.testing.assert_array_equal(S.p, 2.0 * np.ones(S.n_p))
    y_variance = var(S.nat_y)
    if y_variance > 0:
        expected_pen_val = S.n * log(y_variance) + 1e4
    else:
        expected_pen_val = S.n * y_variance + 1e4
    assert S.pen_val == expected_pen_val
    assert S.Psi.shape == (S.n, S.n)
    assert S.psi.shape == (S.n, 1)
    assert S.one.shape == (S.n,)
    assert S.negLnLike is None
    assert S.LnDetPsi is None
    assert S.mu is None
    assert S.U is None
    assert S.SigmaSqr is None
    assert S.Lambda is None

def test_initialize_matrices_adjust_n_p():
    # Create a Kriging instance with n_p greater than k
    S = Kriging(seed=124, n_theta=2, n_p=3, optim_p=True, noise=True)

    # Initialize variables
    nat_X = np.array([[1, 2], [3, 4], [5, 6]])
    nat_y = np.array([1, 2, 3])
    S._initialize_variables(nat_X, nat_y)
    S._set_variable_types()
    S._set_theta_values()

    # Initialize matrices
    S._initialize_matrices()

    # Check if n_p is adjusted to k and matrices are initialized correctly
    assert S.n_p == S.k
    np.testing.assert_array_equal(S.p, 2.0 * np.ones(S.k))
    y_variance = var(S.nat_y)
    if y_variance > 0:
        expected_pen_val = S.n * log(y_variance) + 1e4
    else:
        expected_pen_val = S.n * y_variance + 1e4
    assert S.pen_val == expected_pen_val
    assert S.Psi.shape == (S.n, S.n)
    assert S.psi.shape == (S.n, 1)
    assert S.one.shape == (S.n,)
    assert S.negLnLike is None
    assert S.LnDetPsi is None
    assert S.mu is None
    assert S.U is None
    assert S.SigmaSqr is None
    assert S.Lambda is None

if __name__ == "__main__":
    pytest.main()