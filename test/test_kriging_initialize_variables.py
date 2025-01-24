import pytest
import numpy as np
from spotpython.build import Kriging

def test_initialize_variables():
    # Create a Kriging instance
    S = Kriging()

    # Test with valid input
    nat_X = np.array([[1, 2], [3, 4], [1, 2]])
    nat_y = np.array([1, 2, 11])
    S._initialize_variables(nat_X, nat_y)

    # Check if the instance variables are initialized correctly
    np.testing.assert_array_equal(S.nat_X, nat_X)
    np.testing.assert_array_equal(S.nat_y, nat_y)
    assert S.n == 3
    assert S.k == 2
    np.testing.assert_array_equal(S.min_X, np.array([1, 2]))
    np.testing.assert_array_equal(S.max_X, np.array([3, 4]))
    np.testing.assert_array_equal(S.aggregated_mean_y, np.array([6.0, 2.0]))

    # Test with invalid input dimensions
    nat_X_invalid = np.array([1, 2, 3])
    nat_y_invalid = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="nat_X must be a 2D array and nat_y must be a 1D array."):
        S._initialize_variables(nat_X_invalid, nat_y_invalid)

    # Test with mismatched number of samples
    nat_X_mismatch = np.array([[1, 2], [3, 4]])
    nat_y_mismatch = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="The number of samples in nat_X and nat_y must be equal."):
        S._initialize_variables(nat_X_mismatch, nat_y_mismatch)

if __name__ == "__main__":
    pytest.main()