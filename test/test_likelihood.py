import pytest
import numpy as np
from spotpython.gp.likelihood import gradnlsep

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