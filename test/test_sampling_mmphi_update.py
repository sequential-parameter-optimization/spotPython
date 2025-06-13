import numpy as np
import pytest
from spotpython.utils.sampling import mmphi_intensive_update

def test_mmphi_intensive_update_basic():
    # 2 points in 2D
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    new_point = np.array([0.0, 1.0])
    # Initial distances and multiplicities (only one pair: distance 1.0)
    d = np.array([1.0])
    J = np.array([1])
    q = 2.0
    p = 2.0

    intensive_phiq, updated_J, updated_d = mmphi_intensive_update(X, new_point, J, d, q, p)

    # There are now 3 points, so 3 pairs: (0,1), (0,2), (1,2)
    # Distances: (0,1): 1.0, (0,2): 1.0, (1,2): sqrt(2)
    expected_d = np.array([1.0, np.sqrt(2)])
    expected_J = np.array([2, 1])
    assert np.allclose(np.sort(updated_d), np.sort(expected_d))
    assert np.sum(updated_J) == 3  # 3 pairs

    # Check the value is finite and positive
    assert intensive_phiq > 0
    assert np.isfinite(intensive_phiq)

def test_mmphi_intensive_update_single_point_raises():
    X = np.empty((0, 2))
    new_point = np.array([0.0, 0.0])
    d = np.array([])
    J = np.array([])
    with pytest.raises(ValueError):
        mmphi_intensive_update(X, new_point, J, d)

def test_mmphi_intensive_update_duplicate_distances():
    """
    Test mmphi_intensive_update with duplicate distances.
    """
    X = np.array([[0.0], [1.0], [2.0]])
    new_point = np.array([3.0])
    d = np.array([1.0, 2.0])
    J = np.array([2, 1])
    q = 2.0
    p = 2.0

    intensive_phiq, updated_J, updated_d = mmphi_intensive_update(X, new_point, J, d, q, p)

    # Expected distances and counts
    expected_d = np.array([1.0, 2.0, 3.0])
    expected_J = np.array([3, 2, 1])

    assert np.allclose(np.sort(updated_d), np.sort(expected_d))
    assert np.array_equal(updated_J, expected_J)
    assert np.sum(updated_J) == 6  # 4 points, 6 pairs

    # Check the value is finite and positive
    assert intensive_phiq > 0
    assert np.isfinite(intensive_phiq)

def test_mmphi_intensive_update_nondefault_q_p():
    """
    Test mmphi_intensive_update with non-default values of q and p.
    """
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    new_point = np.array([0.0, 2.0])
    d = np.array([np.sqrt(2)])  # Existing distances should remain unchanged
    J = np.array([1])           # Existing counts should remain unchanged
    q = 1.0
    p = 1.0

    intensive_phiq, updated_J, updated_d = mmphi_intensive_update(X, new_point, J, d, q, p)

    # Expected distances and counts
    # Combine original distances with new distances calculated using Manhattan distance
    new_distances = np.array([np.sum(np.abs(X[0] - new_point)), np.sum(np.abs(X[1] - new_point))])
    all_distances = np.concatenate((d, new_distances))
    expected_d, expected_J = np.unique(all_distances, return_counts=True)

    assert np.allclose(np.sort(updated_d), np.sort(expected_d))
    assert np.array_equal(updated_J, expected_J)
    assert np.sum(updated_J) == len(all_distances)  # Total number of pairs

    # Check the value is finite and positive
    assert intensive_phiq > 0
    assert np.isfinite(intensive_phiq)