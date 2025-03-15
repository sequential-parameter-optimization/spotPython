import pytest
import numpy as np
from spotpython.utils.sampling import perturb

def test_perturb():
    """
    Test the perturb function with various scenarios.
    """

    # Test case 1: Basic functionality with a single perturbation
    X_original = np.array([
        [1, 3],
        [2, 4],
        [3, 1],
        [4, 2]
    ])
    X_perturbed = perturb(X_original.copy(), PertNum=1)
    assert X_perturbed.shape == X_original.shape, "Shape of perturbed array should match the original"
    for col in range(X_original.shape[1]):
        assert set(X_perturbed[:, col]) == set(X_original[:, col]), f"Column {col} should remain a permutation"

    # Test case 2: Multiple perturbations
    X_original = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    X_perturbed = perturb(X_original.copy(), PertNum=5)
    assert X_perturbed.shape == X_original.shape, "Shape of perturbed array should match the original"
    for col in range(X_original.shape[1]):
        assert set(X_perturbed[:, col]) == set(X_original[:, col]), f"Column {col} should remain a permutation"

    # Test case 3: No perturbations (PertNum=0)
    X_original = np.array([
        [1, 2],
        [3, 4]
    ])
    X_perturbed = perturb(X_original.copy(), PertNum=0)
    np.testing.assert_array_equal(X_perturbed, X_original, "No perturbations should result in identical array")

    # Test case 4: Single column (k=1)
    X_original = np.array([
        [1],
        [2],
        [3],
        [4]
    ])
    X_perturbed = perturb(X_original.copy(), PertNum=2)
    assert X_perturbed.shape == X_original.shape, "Shape of perturbed array should match the original"
    assert set(X_perturbed[:, 0]) == set(X_original[:, 0]), "Single column should remain a permutation"

    # Test case 5: Single row (n=1)
    X_original = np.array([[1, 2, 3]])
    X_perturbed = perturb(X_original.copy(), PertNum=3)
    np.testing.assert_array_equal(X_perturbed, X_original, "Single row should remain unchanged")

    # Test case 6: Edge case with empty array
    X_original = np.empty((0, 2))
    X_perturbed = perturb(X_original.copy(), PertNum=1)
    np.testing.assert_array_equal(X_perturbed, X_original, "Empty array should remain unchanged")
