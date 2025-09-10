import numpy as np
import pytest
from spotpython.surrogate.kriging import Kriging


def test_build_psi_vec_basic():
    """Test basic functionality of build_psi_vec"""
    # Setup simple test case
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])
    
    model = Kriging()
    model.fit(X, y)
    
    # Test point
    x_test = np.array([0.5, 0.5])
    psi = model.build_psi_vec(x_test)
    
    # Check shape and basic properties
    assert psi.shape == (2,)
    assert np.all(psi >= 0) and np.all(psi <= 1)


def test_build_psi_vec_isotropic():
    """Test build_psi_vec with isotropic model"""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0])
    
    model = Kriging(isotropic=True)
    model.fit(X, y)
    
    # Points with same distance should have same correlation
    x_test1 = np.array([0.5, 0.0])  # Distance 0.5 from [0,0]
    x_test2 = np.array([0.0, 0.5])  # Distance 0.5 from [0,0]
    
    psi1 = model.build_psi_vec(x_test1)
    psi2 = model.build_psi_vec(x_test2)
    
    assert np.allclose(psi1, psi2)


def test_build_psi_vec_with_categorical():
    """Test build_psi_vec with mixed numeric and categorical variables"""
    X = np.array([[0.0, 0], [1.0, 0], [0.0, 1]])
    y = np.array([0.0, 1.0, 2.0])
    
    model = Kriging(var_type=["num", "factor"])
    model.fit(X, y)
    
    # Test points with same/different categories
    x_same_cat = np.array([0.5, 0])    # Same category as first point
    x_diff_cat = np.array([0.5, 1])    # Different category
    
    psi_same = model.build_psi_vec(x_same_cat)
    psi_diff = model.build_psi_vec(x_diff_cat)
    
    # Correlation should be higher for same category
    assert psi_same[0] > psi_diff[0]


def test_build_psi_vec_extreme_distances():
    """Test build_psi_vec with extreme distances"""
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])
    
    model = Kriging()
    model.fit(X, y)
    
    # Test very close point
    x_close = np.array([0.0, 0.0])  # Same as first training point
    psi_close = model.build_psi_vec(x_close)
    assert np.isclose(psi_close[0], 1.0, atol=1e-10)
    
    # Test very far point
    x_far = np.array([100.0, 100.0])
    psi_far = model.build_psi_vec(x_far)
    assert np.all(psi_far < 0.1)  # Should have low correlation


def test_build_psi_vec_numerical_stability():
    """Test numerical stability of build_psi_vec"""
    X = np.array([[0.0, 0.0], [1e-10, 1e-10]])
    y = np.array([0.0, 1.0])
    
    model = Kriging(nugget=1e-10)
    model.fit(X, y)
    
    # Test with very small distances
    x_test = np.array([1e-20, 1e-20])
    psi = model.build_psi_vec(x_test)
    
    assert not np.any(np.isnan(psi))
    assert not np.any(np.isinf(psi))


def test_build_psi_vec_different_scales():
    """Test build_psi_vec with variables on different scales"""
    X = np.array([[0.0, 0.0], [0.1, 1000.0]])
    y = np.array([0.0, 1.0])
    
    model = Kriging()
    model.fit(X, y)
    
    x_test = np.array([0.05, 500.0])
    psi = model.build_psi_vec(x_test)
    
    assert not np.any(np.isnan(psi))
    assert np.all(psi >= 0) and np.all(psi <= 1)