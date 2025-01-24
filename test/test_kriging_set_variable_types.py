import pytest
import numpy as np
from spotpython.build import Kriging

def test_set_variable_types_valid():
    # Create a Kriging instance with valid var_type
    var_type = ["num", "int", "float"]
    S = Kriging(var_type=var_type, seed=124, n_theta=2, n_p=2, optim_p=True, noise=True)

    # Initialize variables
    nat_X = np.array([[1, 2], [3, 4], [5, 6]])
    nat_y = np.array([1, 2, 3])
    S._initialize_variables(nat_X, nat_y)

    # Set variable types
    S._set_variable_types()

    # Check if the variable types and masks are set correctly
    assert S.var_type == ["num", "int", "float"]
    assert np.all(S.num_mask == np.array([True, False, False]))
    assert np.all(S.int_mask == np.array([False, True, False]))
    assert np.all(S.ordered_mask == np.array([True, True, True]))
    assert np.all(S.factor_mask == np.array([False, False, False]))

def test_set_variable_types_default():
    # Create a Kriging instance with var_type shorter than k
    var_type = ["num"]
    S = Kriging(var_type=var_type, seed=124, n_theta=2, n_p=2, optim_p=True, noise=True)

    # Initialize variables (k = 2, because the dim is 2)
    nat_X = np.array([[1, 2], [3, 4], [5, 6]])
    nat_y = np.array([1, 2, 3])
    S._initialize_variables(nat_X, nat_y)

    # Set variable types
    S._set_variable_types()

    print(f"S.var_type: {S.var_type}")

    # Check if the variable types are defaulted to 'num' and masks are set correctly
    assert S.var_type == ["num", "num"]
    assert np.all(S.num_mask == np.array([True, True]))
    assert np.all(S.int_mask == np.array([False, False]))
    assert np.all(S.ordered_mask == np.array([True, True]))
    assert np.all(S.factor_mask == np.array([False, False]))

def test_set_variable_types_mixed():
    # Create a Kriging instance with mixed var_type
    var_type = ["num", "factor", "int"]
    S = Kriging(var_type=var_type, seed=124, n_theta=2, n_p=2, optim_p=True, noise=True)

    # Initialize variables
    nat_X = np.array([[1, 2], [3, 4], [5, 6]])
    nat_y = np.array([1, 2, 3])
    S._initialize_variables(nat_X, nat_y)

    # Set variable types
    S._set_variable_types()

    # Check if the variable types and masks are set correctly
    assert S.var_type == ["num", "factor", "int"]
    assert np.all(S.num_mask == np.array([True, False, False]))
    assert np.all(S.factor_mask == np.array([False, True, False]))
    assert np.all(S.int_mask == np.array([False, False, True]))
    assert np.all(S.ordered_mask == np.array([True, False, True]))

if __name__ == "__main__":
    pytest.main()