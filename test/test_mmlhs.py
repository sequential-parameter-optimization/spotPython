import pytest
import numpy as np
from spotpython.utils.sampling import mmlhs

def test_mmlhs():
    """
    Test the mmlhs function with various scenarios.
    """

    # Test case 1: Basic functionality with small population and iterations
    X_start = np.array([
        [0, 0],
        [1, 3],
        [2, 1],
        [3, 2]
    ])
    population = 5
    iterations = 10
    X_opt = mmlhs(X_start, population=population, iterations=iterations, q=2.0)
    assert X_opt.shape == X_start.shape, "Optimized plan should have the same shape as the initial plan"
    for col in range(X_start.shape[1]):
        assert set(X_opt[:, col]) == set(X_start[:, col]), f"Column {col} should remain a permutation"

    # Test case 2: Single point (n=1)
    X_start = np.array([[0]])
    population = 3
    iterations = 5
    with pytest.raises(ValueError):
        X_opt = mmlhs(X_start, population=population, iterations=iterations, q=2.0)

    # Test case 3: Single dimension (k=1)
    X_start = np.array([
        [0],
        [1],
        [2],
        [3]
    ])
    population = 4
    iterations = 15
    # check if a ValueError is raised:
    with pytest.raises(ValueError):
        X_opt = mmlhs(X_start, population=population, iterations=iterations, q=2.0)
    # Test case 4: Higher dimensions
    X_start = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]
    ])
    population = 10
    iterations = 20
    X_opt = mmlhs(X_start, population=population, iterations=iterations, q=2.0)
    assert X_opt.shape == X_start.shape, "Optimized plan should have the same shape as the initial plan"
    for col in range(X_start.shape[1]):
        assert set(X_opt[:, col]) == set(X_start[:, col]), f"Column {col} should remain a permutation"

    # Test case 5: Edge case with empty array
    X_start = np.empty((0, 2))
    population = 5
    iterations = 10
    with pytest.raises(ValueError):
        X_opt = mmlhs(X_start, population=population, iterations=iterations, q=2.0)
    
    # Test case 6: Check with different q values
    X_start = np.array([
        [0, 0],
        [1, 3],
        [2, 1],
        [3, 2]
    ])
    population = 5
    iterations = 10
    X_opt_q1 = mmlhs(X_start, population=population, iterations=iterations, q=1.0)
    X_opt_q3 = mmlhs(X_start, population=population, iterations=iterations, q=3.0)
    assert X_opt_q1.shape == X_start.shape, "Optimized plan with q=1 should have the same shape as the initial plan"
    assert X_opt_q3.shape == X_start.shape, "Optimized plan with q=3 should have the same shape as the initial plan"
    for col in range(X_start.shape[1]):
        assert set(X_opt_q1[:, col]) == set(X_start[:, col]), f"Column {col} should remain a permutation for q=1"
        assert set(X_opt_q3[:, col]) == set(X_start[:, col]), f"Column {col} should remain a permutation for q=3"









