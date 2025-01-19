import pytest
from spotpython.hyperparameters.values import convert_keys

def test_convert_keys():
    # Test case 1: Basic conversion to int and float should succeed
    d = {'a': '1', 'b': '2.0', 'c': '3.5'}
    var_type = ["int", "num", "float"]
    result = convert_keys(d, var_type)
    expected = {'a': 1, 'b': 2, 'c': 3.5}
    assert result == expected
    
    # Test case 2: Conversion to int should raise an error for non-integer strings
    d = {'a': '1.5', 'b': '2', 'c': '3'}
    var_type = ["int", "int", "int"]
    with pytest.raises(ValueError, match="Invalid value for conversion at a: 1.5"):
        convert_keys(d, var_type)

    # Test case 3: Conversion with all "num" type should succeed
    d = {'a': '1', 'b': '2.2', 'c': '3'}
    var_type = ["num", "num", "num"]
    result = convert_keys(d, var_type)
    expected = {'a': 1, 'b': 2.2, 'c': 3}
    assert result == expected

    # Test case 4: Check for correct float conversion with "float" type
    d = {'a': '1.0', 'b': '2.5', 'c': '3.1'}
    var_type = ["float", "float", "float"]
    result = convert_keys(d, var_type)
    expected = {'a': 1.0, 'b': 2.5, 'c': 3.1}
    assert result == expected

    # Test case 5: Handling strings that cannot be converted to numbers
    d = {'a': 'hello', 'b': '2', 'c': '3'}
    var_type = ["int", "float", "num"]
    with pytest.raises(ValueError, match="Invalid value for conversion at a: hello"):
        convert_keys(d, var_type)

if __name__ == "__main__":
    pytest.main()