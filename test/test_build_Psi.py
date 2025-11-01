import numpy as np
import pytest
from scipy.spatial.distance import pdist
from spotpython.surrogate.kriging import Kriging


def _init_model_for_build_psi(X, var_type=None, n_theta=1, theta10=None, metric_factorial="canberra"):
    """
    Helper to initialize a Kriging instance minimally for build_Psi() without running fit().
    """
    model = Kriging(var_type=var_type or ["num"], metric_factorial=metric_factorial)
    X = np.asarray(X, dtype=np.float64)
    model.X_ = X
    model.n, model.k = X.shape

    # Ensure masks are set
    model._set_variable_types()

    # Configure n_theta and theta (in log10 scale)
    model.n_theta = n_theta
    if theta10 is None:
        theta10 = np.ones(1 if n_theta == 1 else model.k, dtype=np.float64)
    theta10 = np.asarray(theta10, dtype=np.float64)
    model.theta = np.log10(theta10)

    return model


def test_build_Psi_ordered_isotropic_upper_triangle_values():
    # 1D numeric, isotropic (n_theta=1), weight=1
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    model = _init_model_for_build_psi(X, var_type=["num"], n_theta=1, theta10=[1.0])

    # Build Psi upper triangle
    Psi_upper = model.build_Psi()

    # Expected weighted sqeuclidean distances (isotropic weight=1):
    # (0,1): 1^2 = 1
    # (0,2): 2^2 = 4
    # (1,2): 1^2 = 1
    e = np.exp
    expected = np.array([
        [0.0, e(-1.0), e(-4.0)],
        [0.0, 0.0,    e(-1.0)],
        [0.0, 0.0,    0.0],
    ])
    assert Psi_upper.shape == (3, 3)
    assert np.allclose(Psi_upper, expected, atol=1e-12)
    assert not model.inf_Psi
    assert np.isfinite(model.cnd_Psi)


def test_build_Psi_ordered_anisotropic_weighting():
    # 2D numeric, anisotropic (n_theta=k), weights = [2.0, 0.5]
    X = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [0.0, 2.0],  # 2
    ], dtype=np.float64)
    model = _init_model_for_build_psi(X, var_type=["num", "num"], n_theta=2, theta10=[2.0, 0.5])

    Psi_upper = model.build_Psi()

    # Weighted sqeuclidean distances:
    # (0,1): [1,0] -> 2*1^2 + 0.5*0^2 = 2
    # (0,2): [0,2] -> 2*0^2 + 0.5*2^2 = 2
    # (1,2): [1,-2] -> 2*1^2 + 0.5*2^2 = 2 + 2 = 4
    e = np.exp
    expected = np.array([
        [0.0, e(-2.0), e(-2.0)],
        [0.0, 0.0,     e(-4.0)],
        [0.0, 0.0,     0.0],
    ])
    assert Psi_upper.shape == (3, 3)
    assert np.allclose(Psi_upper, expected, atol=1e-12)
    assert not model.inf_Psi
    assert np.isfinite(model.cnd_Psi)


def test_build_Psi_with_factor_mask_executes_and_values():
    # Check if SciPy supports 'w' in pdist for 'sqeuclidean' and 'canberra'; skip if not.
    try:
        _ = pdist(np.array([[0.0], [1.0]]), metric="sqeuclidean", w=np.array([1.0]))
        _ = pdist(np.array([[0.0], [1.0]]), metric="canberra", w=np.array([1.0]))
    except TypeError:
        pytest.skip("SciPy version does not support 'w' in pdist for required metrics.")

    # 2D: first column numeric (constant -> zero ordered distance), second column factor [0,1,2]
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
    ], dtype=np.float64)
    # var_type marks the second column as 'factor'
    model = _init_model_for_build_psi(X, var_type=["num", "factor"], n_theta=2, theta10=[1.0, 1.0])

    Psi_upper = model.build_Psi()

    # Ordered contribution is zero (first column constant).
    # Factor contribution using canberra on 1D:
    # (0,1): |0-1| / (0+1) = 1
    # (0,2): |0-2| / (0+2) = 1
    # (1,2): |1-2| / (1+2) = 1/3
    e = np.exp
    expected = np.array([
        [0.0, e(-1.0),    e(-1.0)],
        [0.0, 0.0,        e(-(1.0/3.0))],
        [0.0, 0.0,        0.0],
    ])

    assert Psi_upper.shape == (3, 3)
    # Ensure we got an upper triangular matrix
    assert np.allclose(np.tril(Psi_upper), np.zeros_like(Psi_upper))
    # Check expected values
    assert np.allclose(Psi_upper, expected, atol=1e-12)
    assert not model.inf_Psi
    assert np.isfinite(model.cnd_Psi)