import pytest
import numpy as np
from spotpython.surrogate.functions.forr08a import onevar, branin

def test_onevar_single_input():
    # Test with a single input value
    x = np.array([0.5])
    expected = np.array([0.9093])
    result = onevar(x)
    np.testing.assert_allclose(result, expected, rtol=1e-3)

def test_onevar_multiple_inputs():
    # Test with multiple input values
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected = np.array([3.0272, -0.2104, 0.9093,  -5.9933, 15.8297])
    result = onevar(x)
    np.testing.assert_allclose(result, expected, rtol=1e-3)

def test_onevar_out_of_bounds_low():
    # Test with a value below the valid range
    x = np.array([-0.1])
    with pytest.raises(ValueError, match="Variable outside of range - use x in \\[0, 1\\]"):
        onevar(x)

def test_onevar_out_of_bounds_high():
    # Test with a value above the valid range
    x = np.array([1.1])
    with pytest.raises(ValueError, match="Variable outside of range - use x in \\[0, 1\\]"):
        onevar(x)

def test_onevar_empty_array():
    # Test with an empty array
    x = np.array([])
    expected = np.array([])
    result = onevar(x)
    np.testing.assert_array_equal(result, expected)

def test_onevar_edge_cases():
    # Test with edge cases at the boundaries of the range
    x = np.array([0.0, 1.0])
    expected = np.array([3.0272, 15.8297])
    result = onevar(x)
    np.testing.assert_allclose(result, expected, rtol=1e-3)

def test_branin_single_input():
    # Test with a single input value
    x = np.array([[0.5, 0.5]])
    expected = np.array([26.63])
    result = branin(x)
    np.testing.assert_allclose(result, expected, rtol=1e-3)

def test_branin_multiple_inputs():
    # Test with multiple input values
    x = np.array([[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1.0, 1.0]])
    expected = np.array([308.1291, 34.0028, 26.63, 126.3879, 150.8722])
    result = branin(x)
    np.testing.assert_allclose(result, expected, rtol=1e-3)

def test_branin_out_of_bounds_low():
    # Test with a value below the valid range
    x = np.array([[-0.1, 0.5]])
    with pytest.raises(ValueError, match="Variable outside of range - use x in \\[0, 1\\] for both dimensions."):
        branin(x)

def test_branin_out_of_bounds_high():
    # Test with a value above the valid range
    x = np.array([[0.5, 1.1]])
    with pytest.raises(ValueError, match="Variable outside of range - use x in \\[0, 1\\] for both dimensions."):
        branin(x)

def test_branin_invalid_shape():
    # Test with an invalid input shape
    x = np.array([0.5, 0.5, 0.5])
    with pytest.raises(IndexError):
        branin(x)

def test_branin_edge_cases():
    # Test with edge cases at the boundaries of the range
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    expected = np.array([308.1291, 150.8722])
    result = branin(x)
    np.testing.assert_allclose(result, expected, rtol=1e-3)