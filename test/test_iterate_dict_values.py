import numpy as np
import pytest
from spotpython.hyperparameters.values import iterate_dict_values

def test_iterate_dict_values():
    var_dict = {'a': np.array([1, 3, 5]), 'b': np.array([2, 4, 6])}

    # Convert generator to list for testing
    result = list(iterate_dict_values(var_dict))

    # Expected result
    expected = [
        {'a': 1, 'b': 2},
        {'a': 3, 'b': 4},
        {'a': 5, 'b': 6}
    ]

    assert result == expected

    # Test with empty dictionary
    var_dict = {}
    result = list(iterate_dict_values(var_dict))
    assert result == []

    # Test with single key-value pair
    var_dict = {'a': np.array([1, 2, 3])}
    result = list(iterate_dict_values(var_dict))
    expected = [{'a': 1}, {'a': 2}, {'a': 3}]
    assert result == expected

    # Test with different lengths of arrays (should raise an error)
    var_dict = {'a': np.array([1, 2]), 'b': np.array([3, 4, 5])}
    with pytest.raises(ValueError):
        result = list(iterate_dict_values(var_dict))

if __name__ == "__main__":
    pytest.main()