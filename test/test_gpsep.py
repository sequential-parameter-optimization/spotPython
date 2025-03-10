import pytest
import numpy as np
from spotpython.gp.gp_sep import newGPsep, GPsep, getDs, garg, darg

def test_predict_vals():
    """
    Test the predict() method of the GPsep class.
    Ensures the returned dictionary has the correct shape and keys,
    and optionally checks some simple numeric properties.
    """
    # Create some small toy data
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 0.5])
    
    # Instantiate a new GPsep model
    g_val = 1e-5
    gpsep = newGPsep(X, y, d=2.0, g=g_val)
    
    # New points to predict
    XX = np.array([[0.5], [1.5]])
    
    # Evaluate prediction
    result = gpsep.predict(XX, lite=False, nonug=False, return_full=True)
    
    # Check for required keys
    assert "mean" in result
    assert "Sigma" in result
    assert "df" in result
    assert "llik" in result
    
    # Check array shapes
    assert result["mean"].shape == (XX.shape[0],)
    assert result["Sigma"].shape == (XX.shape[0], XX.shape[0])
    assert result["df"].shape == (1,)
    assert result["llik"].shape == (1,)
    
    # Numeric range checks
    # 1. Ensure 'df' is positive
    assert result["df"][0] > 0, "Degrees of freedom should be > 0"
    
    # 2. Check no NaN or infinite values in 'mean'
    assert np.all(np.isfinite(result["mean"])), "Mean contains Inf or NaN"
    
    # 3. Check that the diagonal of Sigma is non-negative
    diag_sigma = np.diag(result["Sigma"])
    assert np.all(diag_sigma >= 0), "Sigma has negative variance on its diagonal"
    
    # 4. Log-likelihood should be finite
    assert np.isfinite(result["llik"][0]), "Log-likelihood is Inf or NaN"


def test_get_d():
    """
    Test that get_d() returns a copy of the GP's length-scale parameters
    and that it raises an error if d is not allocated.
    """
    # Create some small toy data
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 0.5])
    
    # Instantiate a new GPsep model with a known length-scale
    d_val = 2.0
    g_val = 1e-6
    gpsep = newGPsep(X, y, d=d_val, g=g_val, optimize=False)
    
    # get_d should return a copy of the length-scale array
    returned_d = gpsep.get_d()
    # Check shape and contents
    assert returned_d.shape == (1,), "Length-scale array should match the dimension of X"
    assert returned_d[0] == d_val, "Returned length-scale does not match the original"

    # Check that modifying returned_d does not affect gpsep.d
    returned_d[0] = 999
    assert gpsep.d[0] != 999, "gpsep.d should remain unchanged after modifying returned_d"

def test_get_d_raises_error_when_unallocated():
    """
    Test that get_d() raises a ValueError if the internal d is None.
    """
    # Create an empty GPsep object and forcibly set d to None
    gpsep = GPsep()
    gpsep.d = None  # Forcibly create an unallocated scenario

    with pytest.raises(ValueError) as excinfo:
        _ = gpsep.get_d()
    assert "Lengthscale parameter d is not allocated" in str(excinfo.value)
    

def test_getDs_basic():
    """Test the getDs method on a simple 1D dataset."""
    # Create small data
    X = np.linspace(0, 10, 11).reshape(-1, 1)

    # Call getDs
    results = getDs(X=X, p=0.1)

    # Check the result structure
    assert isinstance(results, dict), "getDs should return a dictionary"
    assert "start" in results and "min" in results and "max" in results, \
        "getDs dictionary must contain 'start', 'min', and 'max'"

    # Check they are numeric
    for key in ["start", "min", "max"]:
        assert isinstance(results[key], float), f"{key} should be a float"
        

def test_darg_none():
    """
    Test the darg method when d=None (should default to a dict
    with 'mle' set to True, and fill in values with getDs).
    """
    # Create a small X and y
    X = np.linspace(0, 10, 11).reshape(-1, 1)
 
    # Call darg with d=None
    result = darg(d=None, X=X)

    # Check basic fields
    assert isinstance(result, dict), "Expected darg to return a dict"
    for fld in ["start", "min", "max", "mle", "ab"]:
        assert fld in result, f"Missing expected field '{fld}' in darg result"

    # Check that 'mle' is True by default
    assert result["mle"] is True, "Expected 'mle' to default to True"

    # Check that 'start', 'min', 'max' are floats
    assert np.isscalar(result["start"]), "'start' should be a scalar float"
    assert np.isscalar(result["min"]),   "'min' should be a scalar float"
    assert np.isscalar(result["max"]),   "'max' should be a scalar float"

    # 'ab' should be a 2-vector
    assert len(result["ab"]) == 2, "'ab' should have length 2"


def test_darg_numeric():
    """
    Test the darg method when d is a numeric value.
    Should convert to {"start": d, "mle": True, ...} and 
    fill in any missing fields.
    """
    # Create a small X and y
    X = np.linspace(0, 5, 6).reshape(-1, 1)    
    
    # Pass a numeric value for d
    numeric_d = 2.5
    # Set samp_size >= len(X) to avoid sub-sampling 
    result = darg(d=numeric_d, X=X)

    assert isinstance(result, dict), "Expected a dict result"
    assert "start" in result, "Expected 'start' in the result"
    assert result["start"] == numeric_d, "Numeric input should become d['start']"
    assert "mle" in result and result["mle"] is True, "Expected 'mle' to default to True"
    assert "ab" in result, "Should define 'ab' if 'mle' is True"
    

def test_garg_None():
    """
    Test the garg method when g=None (should default to a dict
    with 'mle' set to False).
    """
    # Create a small vector y
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
   
    # Call garg with g=None
    result = garg(g=None, y=y)
    
    # Check basic fields
    assert isinstance(result, dict), "Expected garg to return a dict"
    for fld in ["start", "min", "max", "mle", "ab"]:
        assert fld in result, f"Missing expected field '{fld}' in garg result"
    
    # Check that 'mle' is False by default
    assert result["mle"] is False, "Expected 'mle' to default to False"
    
    # Check that 'start', 'min', 'max' are floats
    assert np.isscalar(result["start"]), "'start' should be a scalar float"
    assert np.isscalar(result["min"]),   "'min' should be a scalar float"
    assert np.isscalar(result["max"]),   "'max' should be a scalar float"
    
    # 'ab' should be a 2-vector of zeros when mle=False
    assert len(result["ab"]) == 2, "'ab' should have length 2"
    assert all(ab == 0.0 for ab in result["ab"]), "'ab' should be [0.0, 0.0] when mle=False"

def test_garg_numeric():
    """
    Test the garg method when g is a numeric value.
    Should convert to {"start": g, "mle": False, ...} and 
    fill in any missing fields.
    """
    # Create a small vector y
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
   
    # Pass a numeric value for g
    numeric_g = 0.01
    result = garg(g=numeric_g, y=y)
    
    # Check the returned dictionary
    assert isinstance(result, dict), "Expected a dict result"
    assert "start" in result, "Expected 'start' in the result"
    assert result["start"] == numeric_g, "Numeric input should become g['start']"
    assert "mle" in result and result["mle"] is False, "Expected 'mle' to default to False"
    assert "ab" in result and all(ab == 0.0 for ab in result["ab"]), "Should have ab=[0.0, 0.0] when mle=False"

def test_garg_with_mle():
    """
    Test the garg method with mle=True.
    Should set up priors using ab.
    """
    # Create a small vector y with some variance
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
   
   
    # Create a dictionary with mle=True
    g_dict = {"mle": True}
    result = garg(g=g_dict, y=y)
    
    # Check basic fields
    assert isinstance(result, dict), "Expected garg to return a dict"
    assert result["mle"] is True, "'mle' should remain True"
    
    # Check that ab is set properly for mle=True
    assert len(result["ab"]) == 2, "'ab' should have length 2"
    assert result["ab"][0] == 1.5, "'ab[0]' should default to 1.5 for mle=True"
    assert result["ab"][1] > 0, "'ab[1]' should be positive for mle=True"

    # Check that start and max are reasonable
    r2s = (y - np.mean(y))**2
    assert result["start"] <= np.max(r2s), "'start' should be <= max(r2s)"
    assert result["max"] == np.max(r2s), "'max' should equal max(r2s) for mle=True"

def test_garg_errors():
    """
    Test error handling in the garg method.
    """
    # Create a small vector y
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  
    # Test with invalid g type
    with pytest.raises(ValueError):
        garg(g="not_a_dict_or_number", y=y)
    
    # Test with empty y
    with pytest.raises(ValueError):
        garg(g=None, y=np.array([]))