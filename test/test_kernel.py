import numpy as np
import pytest
from spotpython.surrogate.kriging import Kriging # Assuming Kriging class is in this path

# Create a dummy Kriging instance for testing, as _kernel is a method of the class
# We can use default parameters for the Kriging constructor as they don't affect _kernel
@pytest.fixture
def kriging_model():
    """Fixture to create a Kriging model instance."""
    return Kriging()

def test_kernel_simple_1d(kriging_model):
    """Test _kernel with 2 samples, 1 feature."""
    X = np.array([[0.0], [1.0]])
    theta = np.array([1.0])
    p = 2.0
    
    expected_psi_upper_triangle = np.array([
        [0.0, np.exp(-1.0)],
        [0.0, 0.0]
    ])
    
    result = kriging_model._kernel(X, theta, p)
    np.testing.assert_array_almost_equal(result, expected_psi_upper_triangle)

def test_kernel_simple_2d(kriging_model):
    """Test _kernel with 2 samples, 2 features."""
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    theta = np.array([1.0, 1.0])
    p = 2.0
    
    # diff for [0,1]: [abs(0-1)^2, abs(0-1)^2] = [1,1]
    # dist_matrix[0,1] = sum(theta * diff) = 1*1 + 1*1 = 2
    # Psi[0,1] = exp(-2)
    expected_psi_upper_triangle = np.array([
        [0.0, np.exp(-2.0)],
        [0.0, 0.0]
    ])
    
    result = kriging_model._kernel(X, theta, p)
    np.testing.assert_array_almost_equal(result, expected_psi_upper_triangle)

def test_kernel_three_samples_1d(kriging_model):
    """Test _kernel with 3 samples, 1 feature."""
    X = np.array([[0.0], [1.0], [2.0]])
    theta = np.array([0.5])
    p = 2.0
    
    # For X[0] and X[1]: diff = abs(0-1)^2 = 1. dist = 0.5 * 1 = 0.5. val = exp(-0.5)
    # For X[0] and X[2]: diff = abs(0-2)^2 = 4. dist = 0.5 * 4 = 2.0. val = exp(-2.0)
    # For X[1] and X[2]: diff = abs(1-2)^2 = 1. dist = 0.5 * 1 = 0.5. val = exp(-0.5)
    expected_psi_upper_triangle = np.array([
        [0.0, np.exp(-0.5), np.exp(-2.0)],
        [0.0, 0.0,         np.exp(-0.5)],
        [0.0, 0.0,         0.0]
    ])
    
    result = kriging_model._kernel(X, theta, p)
    np.testing.assert_array_almost_equal(result, expected_psi_upper_triangle)

def test_kernel_different_p_value(kriging_model):
    """Test _kernel with a different p value."""
    X = np.array([[0.0], [1.0]])
    theta = np.array([1.0])
    p = 1.0 # Changed p value
    
    # diff for [0,1]: abs(0-1)^1 = 1
    # dist_matrix[0,1] = sum(theta * diff) = 1*1 = 1
    # Psi[0,1] = exp(-1)
    expected_psi_upper_triangle = np.array([
        [0.0, np.exp(-1.0)],
        [0.0, 0.0]
    ])
    
    result = kriging_model._kernel(X, theta, p)
    np.testing.assert_array_almost_equal(result, expected_psi_upper_triangle)

def test_kernel_theta_zeros(kriging_model):
    """Test _kernel when a theta value is zero."""
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    theta = np.array([1.0, 0.0]) # Second feature's theta is 0
    p = 2.0
    
    # diff for [0,1]: [abs(0-1)^2, abs(0-1)^2] = [1,1]
    # dist_matrix[0,1] = sum(theta * diff) = 1*1 + 0*1 = 1
    # Psi[0,1] = exp(-1)
    expected_psi_upper_triangle = np.array([
        [0.0, np.exp(-1.0)],
        [0.0, 0.0]
    ])
    
    result = kriging_model._kernel(X, theta, p)
    np.testing.assert_array_almost_equal(result, expected_psi_upper_triangle)

def test_kernel_no_samples(kriging_model):
    """Test _kernel with no samples. Should produce an empty array or handle gracefully."""
    X = np.empty((0, 2)) # No samples, 2 features
    theta = np.array([1.0, 1.0])
    p = 2.0
    
    # Expected: an empty (0,0) array for the upper triangle
    expected_psi_upper_triangle = np.empty((0,0)) 
    
    result = kriging_model._kernel(X, theta, p)
    # The current implementation of _kernel will likely raise an error or behave unexpectedly
    # with X.shape[0] == 0 before np.triu.
    # np.zeros((0,0)) is valid.
    # np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :]) will result in (0,0,2) shape
    # np.sum on axis 2 will result in (0,0) shape
    # np.exp will result in (0,0) shape
    # np.triu will result in (0,0) shape
    np.testing.assert_array_almost_equal(result, expected_psi_upper_triangle)

def test_kernel_one_sample(kriging_model):
    """Test _kernel with only one sample."""
    X = np.array([[1.0, 2.0]]) # One sample, 2 features
    theta = np.array([1.0, 1.0])
    p = 2.0
    
    # Expected: a (1,1) array with 0 in the upper triangle (as k=1)
    expected_psi_upper_triangle = np.array([[0.0]])
    
    result = kriging_model._kernel(X, theta, p)
    np.testing.assert_array_almost_equal(result, expected_psi_upper_triangle)