import pytest
import numpy as np
from spotpython.utils.sampling import jd

def test_jd():
    """
    Test the jd function with various scenarios.
    """

    # Test case 1: Basic functionality with Euclidean norm (p=2)
    X = np.array([[0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 2.0]])
    J, distinct_d = jd(X, p=2.0)
    expected_distances = np.array([np.sqrt(2), 2 * np.sqrt(2)])
    expected_counts = np.array([2, 1])  # sqrt(2) occurs twice, 2*sqrt(2) occurs once
    np.testing.assert_almost_equal(distinct_d, expected_distances, decimal=7)
    np.testing.assert_array_equal(J, expected_counts)

    # Test case 2: Basic functionality with Manhattan norm (p=1)
    J, distinct_d = jd(X, p=1.0)
    expected_distances = np.array([2.0, 4.0])  # Manhattan distances
    expected_counts = np.array([2, 1])  # 2.0 occurs twice, 4.0 occurs once
    np.testing.assert_almost_equal(distinct_d, expected_distances, decimal=7)
    np.testing.assert_array_equal(J, expected_counts)

    # Test case 3: Single point (no distances)
    X_single = np.array([[0.0, 0.0]])
    J, distinct_d = jd(X_single, p=2.0)
    assert len(distinct_d) == 0, "There should be no distances for a single point"
    assert len(J) == 0, "There should be no multiplicities for a single point"

    # Test case 4: Two points
    X_two = np.array([[0.0, 0.0],
                        [3.0, 4.0]])
    J, distinct_d = jd(X_two, p=2.0)
    expected_distances = np.array([5.0])  # Euclidean distance
    expected_counts = np.array([1])  # Only one distance
    np.testing.assert_almost_equal(distinct_d, expected_distances, decimal=7)
    np.testing.assert_array_equal(J, expected_counts)

    # Test case 5: Higher dimensions
    X_high_dim = np.array([[0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0]])
    J, distinct_d = jd(X_high_dim, p=2.0)
    expected_distances = np.array([np.sqrt(3), 2 * np.sqrt(3)])
    expected_counts = np.array([2, 1])  # sqrt(3) occurs twice, 2*sqrt(3) occurs once
    np.testing.assert_almost_equal(distinct_d, expected_distances, decimal=7)
    np.testing.assert_array_equal(J, expected_counts)

    # Test case 6: Check with p=inf (Chebyshev distance)
    X_inf = np.array([[0.0, 0.0],
                      [1.0, 3.0],
                      [4.0, 1.0]])
    J, distinct_d = jd(X_inf, p=np.inf)
    # Correct distances: [3, 4, 3] => distinct_d = [3, 4], multiplicities J = [2, 1]
    expected_distances = np.array([3.0, 4.0])
    expected_counts = np.array([2, 1])
    np.testing.assert_almost_equal(distinct_d, expected_distances, decimal=7)
    np.testing.assert_array_equal(J, expected_counts)








