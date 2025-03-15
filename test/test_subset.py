import pytest
import numpy as np
from spotpython.utils.sampling import subset

def test_subset():
    """
    Test the subset function with various scenarios.
    """

    # Test case 1: Basic functionality with a 5-point plan in 2D
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.5, 0.5]
    ])
    ns = 3
    Xs, Xr = subset(X, ns=ns)
    assert Xs.shape == (ns, X.shape[1]), f"Subset should have shape ({ns}, {X.shape[1]})"
    assert Xr.shape == (X.shape[0] - ns, X.shape[1]), f"Remainder should have shape ({X.shape[0] - ns}, {X.shape[1]})"
    assert np.all(np.isin(Xs, X)), "Subset points should be part of the original sampling plan"
    assert np.all(np.isin(Xr, X)), "Remainder points should be part of the original sampling plan"
    assert len(np.unique(np.vstack((Xs, Xr)), axis=0)) == X.shape[0], "Subset and remainder should not overlap"

    # Test case 2: Subset size equal to the total number of points
    ns = X.shape[0]
    Xs, Xr = subset(X, ns=ns)
    assert Xs.shape == (ns, X.shape[1]), f"Subset should have shape ({ns}, {X.shape[1]})"
    assert Xr.shape == (0, X.shape[1]), "Remainder should be empty when subset size equals total points"
    np.testing.assert_array_equal(np.sort(Xs, axis=0), np.sort(X, axis=0), "Subset should contain all points")

    # Test case 3: Subset size of 1
    ns = 1
    Xs, Xr = subset(X, ns=ns)
    assert Xs.shape == (ns, X.shape[1]), f"Subset should have shape ({ns}, {X.shape[1]})"
    assert Xr.shape == (X.shape[0] - ns, X.shape[1]), f"Remainder should have shape ({X.shape[0] - ns}, {X.shape[1]})"
    assert np.all(np.isin(Xs, X)), "Subset points should be part of the original sampling plan"
    assert np.all(np.isin(Xr, X)), "Remainder points should be part of the original sampling plan"

    # Test case 4: Higher dimensions
    X_high_dim = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0]
    ])
    ns = 2
    Xs, Xr = subset(X_high_dim, ns=ns)
    assert Xs.shape == (ns, X_high_dim.shape[1]), f"Subset should have shape ({ns}, {X_high_dim.shape[1]})"
    assert Xr.shape == (X_high_dim.shape[0] - ns, X_high_dim.shape[1]), f"Remainder should have shape ({X_high_dim.shape[0] - ns}, {X_high_dim.shape[1]})"
    assert np.all(np.isin(Xs, X_high_dim)), "Subset points should be part of the original sampling plan"
    assert np.all(np.isin(Xr, X_high_dim)), "Remainder points should be part of the original sampling plan"

    # Test case 5: Edge case with empty sampling plan
    X_empty = np.empty((0, 2))
    ns = 0
    Xs, Xr = subset(X_empty, ns=ns)
    assert Xs.shape == (0, 2), "Subset should be empty for an empty sampling plan"
    assert Xr.shape == (0, 2), "Remainder should be empty for an empty sampling plan"

    # Test case 6: Subset size larger than the total number of points
    ns = X.shape[0] + 1
    with pytest.raises(IndexError):
        subset(X, ns=ns)










