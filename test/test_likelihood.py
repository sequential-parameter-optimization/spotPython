import pytest
import numpy as np
from spotpython.gp.likelihood import gradnlsep, nlsep

def test_gradnlsep_basic():
    # Small synthetic example
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    Y = np.array([1.0, 2.0, 3.0])
    par = np.array([0.5, 0.5, 0.1])

    # Call gradnlsep
    grad = gradnlsep(par, X, Y)

    # Check shapes
    assert grad.shape == (3,), "Gradient must have length ncol(X)+1 (=3)."
    # Optional: check for reasonable numeric values
    assert not np.any(np.isnan(grad)), "Gradient should not contain NaNs."


def test_nlsep_inv():
    """
    Test nlsep with chol=False (calls nlsep_0).
    """
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    Y = np.array([1.0, 2.0, 3.0])
    par = np.array([0.5, 0.5, 0.1])

    nll = nlsep(par, X, Y, nlsep_method="inv")
    assert isinstance(nll, float), "nlsep should return a float"
    assert not np.isnan(nll), "Negative log-likelihood should not be NaN"

def test_nlsep_chol():
    """
    Test nlsep with chol=True (calls nlsep_chol).
    """
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    Y = np.array([1.0, 2.0, 3.0])
    par = np.array([0.5, 0.5, 0.1])

    nll = nlsep(par, X, Y, nlsep_method="chol")
    assert isinstance(nll, float), "nlsep should return a float"
    assert not np.isnan(nll), "Negative log-likelihood should not be NaN"

#### python
# filepath: /Users/bartz/workspace/spotPython/test/test_nlsep.py
import pytest
import numpy as np
from spotpython.gp.likelihood import nlsep

def test_nlsep_inv():
    """Test nlsep with method='inv'."""
    X = np.array([[1.0, 2.0], 
                  [3.0, 4.0], 
                  [5.0, 6.0]])
    Y = np.array([1.0, 2.0, 3.0])
    par = np.array([0.5, 0.5, 0.1])
    val = nlsep(par, X, Y, nlsep_method="inv")
    assert isinstance(val, float), "nlsep should return a float."
    assert not np.isnan(val), "Returned value should not be NaN."

def test_nlsep_chol():
    """Test nlsep with method='chol'."""
    X = np.array([[1.0, 2.0], 
                  [3.0, 4.0], 
                  [5.0, 6.0]])
    Y = np.array([1.0, 2.0, 3.0])
    par = np.array([0.5, 0.5, 0.1])
    val = nlsep(par, X, Y, nlsep_method="chol")
    assert isinstance(val, float), "nlsep should return a float."
    assert not np.isnan(val), "Returned value should not be NaN."

def test_nlsep_invalid_method():
    """Test nlsep raises ValueError on invalid method."""
    X = np.array([[1.0, 2.0], 
                  [3.0, 4.0], 
                  [5.0, 6.0]])
    Y = np.array([1.0, 2.0, 3.0])
    par = np.array([0.5, 0.5, 0.1])
    with pytest.raises(ValueError):
        _ = nlsep(par, X, Y, nlsep_method="unknown")
        

def test_gradnlsep_inv():
    """Test gradient with method='inv'."""
    X = np.array([[1.0, 2.0], 
                  [3.0, 4.0], 
                  [5.0, 6.0]])
    Y = np.array([1.0, 2.0, 3.0])
    par = np.array([0.5, 0.5, 0.1])
    grad = gradnlsep(par, X, Y, gradnlsep_method="inv")

    assert grad.shape == (3,), "Gradient should match (ncol(X) + 1)."
    assert not np.any(np.isnan(grad)), "Gradient should not contain NaNs."


def test_gradnlsep_chol():
    """Test gradient with method='chol'."""
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    Y = np.array([1.0, 2.0, 3.0])
    par = np.array([0.2, 1.5, 1e-9])  # small nugget to trigger fallback
    grad = gradnlsep(par, X, Y, gradnlsep_method="chol")

    assert grad.shape == (3,), "Gradient should match (ncol(X) + 1)."
    assert not np.any(np.isnan(grad)), "Gradient should not contain NaNs."


def test_gradnlsep_direct():
    """Test gradient with method='direct' (Cholesky repeated-solve approach)."""
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    Y = np.array([1.0, 2.0, 3.0])
    par = np.array([0.8, 0.4, 0.1])
    grad = gradnlsep(par, X, Y, gradnlsep_method="direct")

    assert grad.shape == (3,), "Gradient should match (ncol(X) + 1)."
    assert not np.any(np.isnan(grad)), "Gradient should not contain NaNs."


def test_gradnlsep_invalid_method():
    """Test that gradnlsep raises ValueError on invalid method string."""
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    Y = np.array([1.0, 2.0, 3.0])
    par = np.array([0.5, 0.5, 0.1])

    with pytest.raises(ValueError):
        _ = gradnlsep(par, X, Y, gradnlsep_method="unknown_method")