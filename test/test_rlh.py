import pytest
import numpy as np
from spotpython.utils.sampling import rlh


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










