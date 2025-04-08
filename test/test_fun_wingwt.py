import numpy as np
from spotpython.fun.objectivefunctions import Analytical


def test_fun_wingwt():
    """Test the fun_wingwt method from the Analytical class."""
    # Create a small test input with shape (2, 10)
    # Here we use the lower and upper bounds of the wing weight function
    X_test = np.array([
    [150, 220,   6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025],
    [200, 300,  10,  10, 45, 1.0, 0.18, 6.0, 2500, 0.08 ],
])
    fun = Analytical()
    result = fun.fun_wingwt(X_test)

    # Check shape of the output
    assert result.shape == (2,), f"Expected output shape (2,), got {result.shape}"

    # Simple check that values are not NaN or inf
    assert np.all(np.isfinite(result)), "Output contains non-finite values."