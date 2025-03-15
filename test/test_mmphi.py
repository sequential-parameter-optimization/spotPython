import pytest
import numpy as np
from spotpython.utils.sampling import mmphi


def test_mmphi():
    """
    Test the mmphi function with various scenarios.
    """

    # Test case 1: Basic functionality with q=2 and p=2 (Euclidean distance)
    X = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0]
    ])
    quality = mmphi(X, q=2, p=2)
    assert quality > 0, "Quality metric should be positive for valid input"

    # Test case 2: Single point (no distances)
    X_single = np.array([[0.0, 0.0]])
    quality_single = mmphi(X_single, q=2, p=2)
    assert quality_single == 0, "Quality metric should be 0 for a single point"

    # Test case 3: Two points
    X_two = np.array([
        [0.0, 0.0],
        [1.0, 1.0]
    ])
    quality_two = mmphi(X_two, q=2, p=2)
    assert quality_two > 0, "Quality metric should be positive for two points"

    # Test case 4: Higher dimensions
    X_high_dim = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0]
    ])
    quality_high_dim = mmphi(X_high_dim, q=2, p=2)
    assert quality_high_dim > 0, "Quality metric should be positive for higher dimensions"

    # Test case 5: Check with Manhattan distance (p=1)
    X_manhattan = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0]
    ])
    quality_manhattan = mmphi(X_manhattan, q=2, p=1)
    assert quality_manhattan > 0, "Quality metric should be positive with Manhattan distance"

    # Test case 6: Edge case with empty sampling plan
    X_empty = np.empty((0, 2))
    quality_empty = mmphi(X_empty, q=2, p=2)
    assert quality_empty == 0, "Quality metric should be 0 for an empty sampling plan"

    # Test case 7: Check with q=1 (different exponent)
    X_q1 = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0]
    ])
    quality_q1 = mmphi(X_q1, q=1, p=2)
    assert quality_q1 > 0, "Quality metric should be positive with q=1"







