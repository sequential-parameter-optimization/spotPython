#### python
# filepath: /Users/bartz/workspace/spotPython/test/test_sampling.py
import pytest
import numpy as np
from spotpython.utils.sampling import fullfactorial, rlh, jd, mm, mmphi

def test_fullfactorial():
    """
    Test the fullfactorial function with various scenarios.
    """

    # Test case 1: Basic functionality with Edges=1 (points equally spaced from edge to edge)
    q = [2, 3]
    expected_output = np.array([
        [0.0, 0.0],
        [0.0, 0.5],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.0, 1.0]
    ])
    np.testing.assert_almost_equal(fullfactorial(q, Edges=1), expected_output, decimal=7)

    # Test case 2: Basic functionality with Edges=0 (points at midpoints of bins)
    # In the first dimension with q[0] = 2, the midpoints are [0.0, 0.75].
    # In the second dimension with q[1] = 3, the midpoints are [0.0, 0.4167, 0.8333].
    actual_output = fullfactorial(q, Edges=0)
    expected_output_edges0 = np.array([
        [0.0, 0.0],
        [0.0, 0.4166667],
        [0.0, 0.8333333],
        [0.75, 0.0],
        [0.75, 0.4166667],
        [0.75, 0.8333333]
    ])
    np.testing.assert_almost_equal(actual_output, expected_output_edges0, decimal=7)

    # Test case 3: Check if ValueError is raised for dimensions with less than 2 points
    q_invalid = [1, 3]
    with pytest.raises(ValueError):
        fullfactorial(q_invalid, Edges=1)

    # Test case 4: Check with a single dimension and multiple levels
    q_single = [5]
    expected_output_single = np.linspace(0, 1, 5).reshape((5, 1))
    np.testing.assert_almost_equal(fullfactorial(q_single, Edges=1), expected_output_single, decimal=7)

    # Test case 5: Verify shape for higher dimensions with Edges=1
    q_higher = [5, 2]
    output = fullfactorial(q_higher, Edges=1)
    assert output.shape == (10, 2), f"Expected shape (10, 2), got {output.shape}"

def test_rlh():
    """
    Test the rlh function with various scenarios.
    """

    # Test case 1: Basic functionality with edges=0
    n, k = 5, 2
    output = rlh(n, k, edges=0)
    assert output.shape == (n, k), f"Expected shape ({n}, {k}), got {output.shape}"
    assert np.all(output >= 0) and np.all(output <= 1), "All values should be within [0, 1]"
    for i in range(k):
        assert len(np.unique(output[:, i])) == n, f"Column {i} should have {n} unique values"

    # Test case 2: Basic functionality with edges=1
    output_edges = rlh(n, k, edges=1)
    assert output_edges.shape == (n, k), f"Expected shape ({n}, {k}), got {output_edges.shape}"
    assert np.all(output_edges >= 0) and np.all(output_edges <= 1), "All values should be within [0, 1]"
    for i in range(k):
        assert len(np.unique(output_edges[:, i])) == n, f"Column {i} should have {n} unique values"

    # Test case 3: Check for single dimension (k=1)
    n, k = 10, 1
    output_single_dim = rlh(n, k, edges=0)
    assert output_single_dim.shape == (n, k), f"Expected shape ({n}, {k}), got {output_single_dim.shape}"
    assert np.all(output_single_dim >= 0) and np.all(output_single_dim <= 1), "All values should be within [0, 1]"
    assert len(np.unique(output_single_dim[:, 0])) == n, "Column 0 should have n unique values"

    # Test case 4: Check for single point (n=1)
    n, k = 1, 3
    output_single_point = rlh(n, k, edges=0)
    assert output_single_point.shape == (n, k), f"Expected shape ({n}, {k}), got {output_single_point.shape}"
    assert np.all(output_single_point >= 0) and np.all(output_single_point <= 1), "All values should be within [0, 1]"

    # Test case 5: Check for higher dimensions
    n, k = 7, 5
    output_higher_dim = rlh(n, k, edges=0)
    assert output_higher_dim.shape == (n, k), f"Expected shape ({n}, {k}), got {output_higher_dim.shape}"
    assert np.all(output_higher_dim >= 0) and np.all(output_higher_dim <= 1), "All values should be within [0, 1]"
    for i in range(k):
        assert len(np.unique(output_higher_dim[:, i])) == n, f"Column {i} should have {n} unique values"

    # Test case 6: Check for edges=1 scaling
    n, k = 4, 2
    output_edges_check = rlh(n, k, edges=1)
    assert np.isclose(output_edges_check.min(), 0.0), "Minimum value should be close to 0.0 for edges=1"
    assert np.isclose(output_edges_check.max(), 1.0), "Maximum value should be close to 1.0 for edges=1"
        
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




