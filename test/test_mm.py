import pytest
import numpy as np
from spotpython.utils.sampling import mm

def test_mm():
    """
    Test the mm function with various scenarios.
    """

    # Test case 1: Identical sampling plans
    X1 = np.array([[0.0, 0.0],
                    [0.5, 0.5],
                    [1.0, 1.0]])
    X2 = np.array([[0.0, 0.0],
                    [0.5, 0.5],
                    [1.0, 1.0]])
    assert mm(X1, X2, p=2.0) == 0, "Identical sampling plans should return 0"

    # Test case 2: X1 is more space-filling than X2
    X1 = np.array([[0.0, 0.0],
                    [0.5, 0.5],
                    [1.0, 1.0]])
    X2 = np.array([[0.0, 0.0],
                    [0.1, 0.1],
                    [0.2, 0.2]])
    assert mm(X1, X2, p=2.0) == 1, "X1 should be more space-filling than X2"

    # Test case 3: X2 is more space-filling than X1
    X1 = np.array([[0.0, 0.0],
                    [0.1, 0.1],
                    [0.2, 0.2]])
    X2 = np.array([[0.0, 0.0],
                    [0.5, 0.5],
                    [1.0, 1.0]])
    assert mm(X1, X2, p=2.0) == 2, "X2 should be more space-filling than X1"

    # Test case 4: Higher dimensions with p=1 (Manhattan distance)
    X1 = np.array([[0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                    [1.0, 1.0, 1.0]])
    X2 = np.array([[0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.1],
                    [0.2, 0.2, 0.2]])
    assert mm(X1, X2, p=1.0) == 1, "X1 should be more space-filling than X2 with Manhattan distance"

    # Test case 5: Single point in each sampling plan
    X1 = np.array([[0.0, 0.0]])
    X2 = np.array([[1.0, 1.0]])
    assert mm(X1, X2, p=2.0) == 0, "Single points should be considered equally space-filling"

    # Test case 6: Edge case with empty sampling plans
    X1 = np.empty((0, 2))
    X2 = np.empty((0, 2))
    assert mm(X1, X2, p=2.0) == 0, "Empty sampling plans should return 0"

