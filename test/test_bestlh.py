import pytest
import numpy as np
from spotpython.utils.sampling import bestlh
def test_bestlh():
    """
    Test the bestlh function with various scenarios.
    """

    # Test case 1: Basic functionality with small population and iterations
    n, k, population, iterations = 3, 2, 5, 10
    X_opt = bestlh(n=n, k=k, population=population, iterations=iterations, p=1, plot=False)
    assert X_opt.shape == (n, k), "Optimized plan should have the correct shape"
    for col in range(k):
        assert len(np.unique(X_opt[:, col])) == n, f"Column {col} should have {n} unique values"

    # Test case 2: Single dimension (k=2 is the minimum allowed)
    n, k, population, iterations = 3, 2, 5, 10
    X_opt = bestlh(n=n, k=k, population=population, iterations=iterations, p=1, plot=False)
    assert X_opt.shape == (n, k), "Optimized plan should have the correct shape"
    for col in range(k):
        assert len(np.unique(X_opt[:, col])) == n, f"Column {col} should have {n} unique values"

    # Test case 3: Higher dimensions
    n, k, population, iterations = 3, 3, 5, 10
    X_opt = bestlh(n=n, k=k, population=population, iterations=iterations, p=2, plot=False)
    assert X_opt.shape == (n, k), "Optimized plan should have the correct shape"
    for col in range(k):
        assert len(np.unique(X_opt[:, col])) == n, f"Column {col} should have {n} unique values"

    # Test case 4: Edge case with minimum points (n=2)
    n, k, population, iterations = 2, 2, 5, 10
    X_opt = bestlh(n=n, k=k, population=population, iterations=iterations, p=1, plot=False)
    assert X_opt.shape == (n, k), "Optimized plan should have the correct shape"
    for col in range(k):
        assert len(np.unique(X_opt[:, col])) == n, f"Column {col} should have {n} unique values"

    # Test case 5: Check with different distance norms (p=1 and p=2)
    n, k, population, iterations = 3, 2, 5, 10
    X_opt_p1 = bestlh(n=n, k=k, population=population, iterations=iterations, p=1, plot=False)
    X_opt_p2 = bestlh(n=n, k=k, population=population, iterations=iterations, p=2, plot=False)
    assert X_opt_p1.shape == (n, k), "Optimized plan with p=1 should have the correct shape"
    assert X_opt_p2.shape == (n, k), "Optimized plan with p=2 should have the correct shape"

    # Test case 6: Check if ValueError is raised for k < 2
    n, k, population, iterations = 3, 1, 5, 10
    with pytest.raises(ValueError):
        bestlh(n=n, k=k, population=population, iterations=iterations, p=1, plot=False)

    # Test case 7: Edge case with empty Latin hypercube
    n, k, population, iterations = 0, 2, 5, 10
    with pytest.raises(ValueError):
        bestlh(n=n, k=k, population=population, iterations=iterations, p=1, plot=False)











