import numpy as np
import pandas as pd
import pytest
from spotpython.utils.effects import screening_print, screening_plot, randorient

def mock_objective_function(x):
    """Mock objective function for testing."""
    return np.sum(x**2)


@pytest.fixture
def test_data():
    """Fixture to provide test data."""
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1]
    ])
    labels = ["x1", "x2"]
    range_ = np.array([[0, 0], [1, 1]])
    return X, labels, range_


def test_screening_dataframe_output(test_data):
    """Test if screening returns a DataFrame with correct structure."""
    X, labels, range_ = test_data
    xi = 1
    p = 3

    result = screening_print(X, mock_objective_function, xi, p, labels, bounds=range_)

    assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"
    assert set(result.columns) == {"varname", "mean", "sd"}, "DataFrame should have columns 'varname', 'mean', and 'sd'"
    assert len(result) == len(labels), "DataFrame should have one row per variable"


def test_screening_mean_and_sd_calculation(test_data):
    """Test if the mean and standard deviation are calculated correctly."""
    X, labels, range_ = test_data
    xi = 1
    p = 3

    result = screening_print(X, mock_objective_function, xi, p, labels, bounds=range_)

    # Check if mean and standard deviation are non-negative
    assert (result["mean"] >= 0).all(), "Mean values should be non-negative"
    assert (result["sd"] >= 0).all(), "Standard deviation values should be non-negative"


def test_screening_with_no_range(test_data):
    """Test screening function when no bounds is provided."""
    X, labels, _ = test_data
    xi = 1
    p = 3

    result = screening_print(X, mock_objective_function, xi, p, labels, bounds=None)

    assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"
    assert len(result) == len(labels), "DataFrame should have one row per variable"


def test_screening_plot_output(test_data):
    """Test if screening generates a plot when print=False."""
    X, labels, range_ = test_data
    xi = 1
    p = 3

    # Ensure no exceptions are raised during plotting
    try:
        screening_plot(X, mock_objective_function, xi, p, labels, bounds=range_, show=False)
    except Exception as e:
        pytest.fail(f"Plotting failed with exception: {e}")
        
def test_randorient_output_shape():
    """Test if randorient returns an array with the correct shape."""
    k = 3
    p = 3
    xi = 1
    result = randorient(k, p, xi)
    assert isinstance(result, np.ndarray), "Output should be a numpy array"
    assert result.shape == (k + 1, k), f"Output shape should be {(k + 1, k)}"


def test_randorient_randomness():
    """Test if randorient produces different outputs with different seeds."""
    k = 3
    p = 3
    xi = 1
    seed1 = 42
    seed2 = 24
    result1 = randorient(k, p, xi, seed=seed1)
    result2 = randorient(k, p, xi, seed=seed2)
    assert not np.array_equal(result1, result2), "Outputs with different seeds should not be identical"


def test_randorient_reproducibility():
    """Test if randorient produces the same output with the same seed."""
    k = 3
    p = 3
    xi = 1
    seed = 42
    result1 = randorient(k, p, xi, seed=seed)
    result2 = randorient(k, p, xi, seed=seed)
    assert np.array_equal(result1, result2), "Outputs with the same seed should be identical"


def test_randorient_step_length():
    """Test if randorient respects the step length."""
    k = 2
    p = 3
    xi = 0.2
    result = randorient(k, p, xi)
    step_lengths = np.diff(result[:, 0])  # Check step lengths along the first dimension
    unique_steps = np.unique(np.abs(step_lengths))[-1]
    # print(f"Unique step lengths: {unique_steps}")
    # Calculate expected step length
    expected_step = xi / (p - 1)
    # print(f"Expected step length: {expected_step}")
    assert np.isclose(unique_steps, expected_step).all(), f"Step length should be approximately {expected_step}"


def test_randorient_invalid_inputs():
    """Test if randorient raises errors for invalid inputs."""
    with pytest.raises(ZeroDivisionError):
        randorient(k=2, p=1, xi=1)  # p=1 would cause division by zero
    with pytest.raises(ValueError):
        randorient(k=-1, p=3, xi=1)  # Negative dimensions are invalid


def test_randorient_result():
    """Test if randorient result."""
    k = 2
    p = 3
    xi = 1
    result = randorient(k, p, xi)
    assert isinstance(result, np.ndarray), "Output should be a numpy array"
    assert result.shape == (k + 1, k), f"Output shape should be {(k + 1, k)}"
    # check if the result contains only 0s and 0.5s
    assert np.all(np.isin(result, [0, 0.5])), "Result should contain only 0s and 0.5s"
    
    
    