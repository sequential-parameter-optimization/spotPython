import pytest
import numpy as np
from spotpython.utils.sampling import fullfactorial
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
