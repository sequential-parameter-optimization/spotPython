import pytest
import numpy as np
from spotpython.hyperparameters.values import assign_values

def test_assign_values():
    # Test with a 2x2 array
    X = np.array([[1, 2], [3, 4]])
    var_list = ['a', 'b']
    result = assign_values(X, var_list)
    expected = {'a': np.array([1, 3]), 'b': np.array([2, 4])}
    assert result.keys() == expected.keys()
    for key in result:
        assert np.array_equal(result[key], expected[key])

    # Test with a 3x2 array
    X = np.array([[1, 2], [3, 4], [5, 6]])
    var_list = ['a', 'b']
    result = assign_values(X, var_list)
    expected = {'a': np.array([1, 3, 5]), 'b': np.array([2, 4, 6])}
    assert result.keys() == expected.keys()
    for key in result:
        assert np.array_equal(result[key], expected[key])

    # Test with a 3x3 array
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    var_list = ['a', 'b', 'c']
    result = assign_values(X, var_list)
    expected = {'a': np.array([1, 4, 7]), 'b': np.array([2, 5, 8]), 'c': np.array([3, 6, 9])}
    assert result.keys() == expected.keys()
    for key in result:
        assert np.array_equal(result[key], expected[key])

    # Test with empty array and empty var_list
    X = np.array([[]])
    var_list = []
    result = assign_values(X, var_list)
    expected = {}
    assert result == expected

    # Test with mismatched var_list length
    X = np.array([[1, 2], [3, 4]])
    var_list = ['a']
    with pytest.raises(ValueError):
        assign_values(X, var_list)

if __name__ == "__main__":
    pytest.main()