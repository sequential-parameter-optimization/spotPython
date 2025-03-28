import numpy as np
import pytest
from spotpython.utils.desirability import DOverall, DMax, DMin, DTarget, DArb, DBox, DCategorical

def test_doverall_initialization():
    """Test initialization of DOverall with valid desirability objects."""
    dmax = DMax(low=0, high=10, scale=1)
    dmin = DMin(low=5, high=15, scale=1)
    doverall = DOverall(dmax, dmin)
    assert len(doverall.d_objs) == 2
    assert isinstance(doverall.d_objs[0], DMax)
    assert isinstance(doverall.d_objs[1], DMin)

def test_doverall_invalid_initialization():
    """Test initialization of DOverall with invalid objects."""
    with pytest.raises(ValueError, match="All objects must be instances of valid desirability classes."):
        DOverall(DMax(low=0, high=10, scale=1), "invalid_object")

def test_doverall_predict():
    """Test the predict method of DOverall."""
    dmax = DMax(low=0, high=10, scale=1)
    dmin = DMin(low=5, high=15, scale=1)
    doverall = DOverall(dmax, dmin)

    inputs = np.array([[5, 10], [0, 15], [10, 5]])
    overall_desirability = doverall.predict(inputs)
    assert overall_desirability.shape == (3,)
    assert np.all(overall_desirability >= 0) and np.all(overall_desirability <= 1)

def test_doverall_predict_all():
    """Test the predict method of DOverall with all=True."""
    dmax = DMax(low=0, high=10, scale=1)
    dmin = DMin(low=5, high=15, scale=1)
    doverall = DOverall(dmax, dmin)

    inputs = np.array([[5, 10], [0, 15], [10, 5]])
    individual, overall = doverall.predict(inputs, all=True)
    assert len(individual) == 2  # Two desirability objects
    assert individual[0].shape == (3,)
    assert individual[1].shape == (3,)
    assert overall.shape == (3,)
    assert np.all(overall >= 0) and np.all(overall <= 1)

def test_doverall_invalid_input_shape():
    dmax = DMax(low=0, high=10, scale=1)
    dmin = DMin(low=5, high=15, scale=1)
    doverall = DOverall(dmax, dmin)

    # This array has shape (3,) => does not match 2 objects
    inputs = np.array([5, 10, 15])
    with pytest.raises(ValueError, match="The number of columns in newdata must match"):
        doverall.predict(inputs)

def test_doverall_with_various_objects():
    """Test DOverall with a mix of desirability objects."""
    dmax = DMax(low=0, high=10, scale=1)
    dmin = DMin(low=5, high=15, scale=1)
    dtarget = DTarget(low=0, target=5, high=10, low_scale=1, high_scale=1)
    doverall = DOverall(dmax, dmin, dtarget)

    inputs = np.array([[5, 10, 5], [0, 15, 0], [10, 5, 10]])
    overall_desirability = doverall.predict(inputs)
    assert overall_desirability.shape == (3,)
    assert np.all(overall_desirability >= 0) and np.all(overall_desirability <= 1)