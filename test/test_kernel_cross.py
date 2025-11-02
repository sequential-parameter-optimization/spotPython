import numpy as np
import pytest

from spotpython.surrogate.kriging import Kriging


def make_model(k=2, var_type=None, theta_log10=None, isotropic=False):
    model = Kriging(var_type=var_type or ["num"] * k, isotropic=isotropic)
    # Minimal setup for _kernel_cross
    model.k = k
    model._set_variable_types()
    if isotropic:
        model.n_theta = 1
        if theta_log10 is None:
            theta_log10 = np.array([0.0])  # 10**0 = 1
    else:
        model.n_theta = k
        if theta_log10 is None:
            theta_log10 = np.zeros(k)  # 10**0 = 1 for all dims
    model.theta = np.asarray(theta_log10, dtype=float)
    return model


def test_kernel_cross_shape_and_values_basic():
    # 2D, numeric only
    model = make_model(k=2)
    A = np.array([[0.0, 0.0], [1.0, 1.0]])
    B = np.array([[0.5, 0.5], [1.5, 1.5], [0.0, 1.0]])
    K = model._kernel_cross(A, B)

    assert K.shape == (A.shape[0], B.shape[0])
    # Values must be in (0, 1], strictly positive and <= 1
    assert np.all(K > 0.0)
    assert np.all(K <= 1.0)


def test_kernel_cross_symmetry_and_diagonal_when_A_equals_B():
    model = make_model(k=3)
    A = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.3, 0.4],
        [1.0, 1.0, 1.0],
    ])
    K = model._kernel_cross(A, A)
    # Symmetric
    assert np.allclose(K, K.T, atol=1e-12)
    # Diagonal must be exactly 1.0
    assert np.allclose(np.diag(K), 1.0)


def test_kernel_cross_monotone_in_theta():
    # Larger theta (log10) => larger weights => smaller correlations off-diagonal
    A = np.array([[0.0, 0.0], [1.0, 1.0]])
    B = np.array([[0.5, 0.5], [1.5, 1.5]])

    model_small = make_model(k=2, theta_log10=np.array([0.0, 0.0]))   # weights = 1,1
    model_large = make_model(k=2, theta_log10=np.array([1.0, 1.0]))   # weights = 10,10

    K_small = model_small._kernel_cross(A, B)
    K_large = model_large._kernel_cross(A, B)

    # Off-diagonal entries should shrink with larger weights
    # Compare all entries except potential exact matches on diagonal (not square here anyway)
    assert np.all(K_large <= K_small + 1e-12)
    assert np.any(K_large < K_small - 1e-12)


def test_kernel_cross_handles_1d_inputs():
    model = make_model(k=2)
    a = np.array([0.0, 0.0])       # 1D A
    B = np.array([[0.0, 0.0], [1.0, 1.0]])
    K = model._kernel_cross(a, B)
    assert K.shape == (1, 2)
    assert np.isclose(K[0, 0], 1.0)  # same point => exp(0) = 1


def test_kernel_cross_isotropic_matches_anisotropic_when_equal_thetas():
    # If anisotropic theta are equal per-dim, isotropic should match
    A = np.array([[0.0, 0.0], [0.5, 0.5]])
    B = np.array([[0.25, 0.25], [1.0, 1.0]])

    theta_equal = np.array([0.3, 0.3])   # per-dimension log10(theta)
    model_aniso = make_model(k=2, theta_log10=theta_equal, isotropic=False)

    # Isotropic uses one theta; set it equal to per-dim value
    model_iso = make_model(k=2, theta_log10=np.array([0.3]), isotropic=True)

    K_aniso = model_aniso._kernel_cross(A, B)
    K_iso = model_iso._kernel_cross(A, B)
    assert np.allclose(K_aniso, K_iso, atol=1e-12)