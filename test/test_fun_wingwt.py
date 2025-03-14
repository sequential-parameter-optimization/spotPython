import numpy as np
from spotpython.fun.objectivefunctions import Analytical


def test_fun_wingwt():
    """Test the fun_wingwt method from the Analytical class."""
    # Create a small test input with shape (n, 10)
    X_test = np.array([
        [0.0]*10,
        [1.0]*10
    ])
    fun = Analytical()
    result = fun.fun_wingwt(X_test)

    # Check shape of the output
    assert result.shape == (2,), f"Expected output shape (2,), got {result.shape}"

    # Simple check that values are not NaN or inf
    assert np.all(np.isfinite(result)), "Output contains non-finite values."