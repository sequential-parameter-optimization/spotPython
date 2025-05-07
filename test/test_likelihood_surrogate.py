import numpy as np
import pytest
from spotpython.surrogate.kriging import Kriging


@pytest.mark.parametrize("method, x_shape", [
    ("interpolation", (1,)),
    ("regression", (2,)),
    ("reinterpolation", (2,))
])
def test_likelihood_basic(method, x_shape):
    # Small toy data: 3 points, 1D
    X = np.array([[0.0], [0.5], [1.0]])
    y = np.array([1.0, 2.0, 3.0])
    model = Kriging(method=method, penalty=12345.0)
    model.X_ = X
    model.y_ = y
    model.eps = 1e-6
    model.penalty = 12345.0

    # log10(theta) = 0, log10(lambda) = -3 for regression/reinterpolation
    if method == "interpolation":
        x = np.zeros(x_shape)
    else:
        x = np.zeros(x_shape)
        x[-1] = -3  # log10(lambda)

    negLnLike, Psi, U = model.likelihood(x)

    # Check types and shapes
    assert isinstance(negLnLike, float)
    assert Psi.shape == (3, 3)
    assert np.allclose(Psi, Psi.T)
    if U is not None:
        assert U.shape == (3, 3)
        # U should be upper-triangular (Cholesky returns lower, but code uses U as upper)
        # Actually, np.linalg.cholesky returns lower-triangular, so U @ U.T == Psi
        assert np.allclose(U @ U.T, Psi, atol=1e-8)
    else:
        # If U is None, penalty should be returned
        assert negLnLike == model.penalty

    # negLnLike should be finite unless penalty
    if U is not None:
        assert np.isfinite(negLnLike)

def test_likelihood_invalid_method():
    X = np.array([[0.0], [1.0]])
    y = np.array([1.0, 2.0])
    model = Kriging(method="regression")
    model.X_ = X
    model.y_ = y
    # Set an invalid method
    model.method = "invalid"
    with pytest.raises(ValueError):
        model.likelihood(np.zeros(1))

def test_likelihood_ill_conditioned_returns_penalty():
    # Use two identical points to force singular Psi
    X = np.array([[0.0], [0.0]])
    y = np.array([1.0, 1.0])
    model = Kriging(method="interpolation", penalty=99999.0)
    model.X_ = X
    model.y_ = y
    model.eps = 0.0  # No regularization
    x = np.zeros(1)
    negLnLike, Psi, U = model.likelihood(x)
    assert negLnLike == 99999.0
    assert U is None
    assert Psi.shape == (2, 2)