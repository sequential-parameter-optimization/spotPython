import pytest
import numpy as np
from spotpython.utils.sampling import phisort
def test_phisort():
    """
    Test the phisort function with various scenarios.
    """

    # Test case 1: Two sampling plans in 2D
    X1 = np.array([[0.0, 0.0],
                    [0.5, 0.5],
                    [1.0, 1.0]])
    X2 = np.array([[0.2, 0.2],
                    [0.6, 0.4],
                    [0.9, 0.9]])
    X3D = np.stack([X1, X2], axis=2)
    ranking = phisort(X3D, q=2.0, p=2.0)
    assert len(ranking) == 2, "Ranking should have 2 elements for 2 sampling plans"
    assert set(ranking) == {1, 2}, "Ranking should contain indices 1 and 2"

    # Test case 2: Three sampling plans in 2D
    X3 = np.array([[0.1, 0.1],
                    [0.4, 0.4],
                    [0.8, 0.8]])
    X3D = np.stack([X1, X2, X3], axis=2)
    ranking = phisort(X3D, q=2.0, p=2.0)
    assert len(ranking) == 3, "Ranking should have 3 elements for 3 sampling plans"
    assert set(ranking) == {1, 2, 3}, "Ranking should contain indices 1, 2, and 3"

    # Test case 3: Single sampling plan
    X3D = np.expand_dims(X1, axis=2)
    ranking = phisort(X3D, q=2.0, p=2.0)
    assert len(ranking) == 1, "Ranking should have 1 element for a single sampling plan"
    assert ranking[0] == 1, "Ranking should be [1] for a single sampling plan"

    # Test case 4: Higher dimensions
    X1 = np.array([[0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                    [1.0, 1.0, 1.0]])
    X2 = np.array([[0.1, 0.1, 0.1],
                    [0.4, 0.4, 0.4],
                    [0.9, 0.9, 0.9]])
    X3D = np.stack([X1, X2], axis=2)
    ranking = phisort(X3D, q=2.0, p=2.0)
    assert len(ranking) == 2, "Ranking should have 2 elements for 2 sampling plans in higher dimensions"
    assert set(ranking) == {1, 2}, "Ranking should contain indices 1 and 2"

    # Test case 5: Edge case with empty sampling plans
    X3D = np.empty((0, 2, 2))
    ranking = phisort(X3D, q=2.0, p=2.0)
    assert len(ranking) == 2, "Ranking should handle empty sampling plans and return indices"

    # Test case 6: Edge case with identical sampling plans
    X3D = np.stack([X1, X1], axis=2)
    ranking = phisort(X3D, q=2.0, p=2.0)
    assert len(ranking) == 2, "Ranking should have 2 elements for identical sampling plans"
    assert set(ranking) == {1, 2}, "Ranking should contain indices 1 and 2 for identical plans"

    # Test case 7: Check with different q values
    X3D = np.stack([X1, X2], axis=2)
    ranking_q1 = phisort(X3D, q=1.0, p=2.0)
    ranking_q3 = phisort(X3D, q=3.0, p=2.0)
    assert len(ranking_q1) == 2, "Ranking with q=1 should have 2 elements"
    assert len(ranking_q3) == 2, "Ranking with q=3 should have 2 elements"
    assert set(ranking_q1) == {1, 2}, "Ranking with q=1 should contain indices 1 and 2"
    assert set(ranking_q3) == {1, 2}, "Ranking with q=3 should contain indices 1 and 2"





