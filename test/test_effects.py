import numpy as np
import pandas as pd
import pytest
from spotpython.utils.effects import screening

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
    xi = 0.1
    p = 3

    result = screening(X, mock_objective_function, xi, p, labels, bounds=range_, print=True)

    assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"
    assert set(result.columns) == {"varname", "mean", "sd"}, "DataFrame should have columns 'varname', 'mean', and 'sd'"
    assert len(result) == len(labels), "DataFrame should have one row per variable"


def test_screening_mean_and_sd_calculation(test_data):
    """Test if the mean and standard deviation are calculated correctly."""
    X, labels, range_ = test_data
    xi = 0.1
    p = 3

    result = screening(X, mock_objective_function, xi, p, labels, bounds=range_, print=True)

    # Check if mean and standard deviation are non-negative
    assert (result["mean"] >= 0).all(), "Mean values should be non-negative"
    assert (result["sd"] >= 0).all(), "Standard deviation values should be non-negative"


def test_screening_with_no_range(test_data):
    """Test screening function when no bounds is provided."""
    X, labels, _ = test_data
    xi = 0.1
    p = 3

    result = screening(X, mock_objective_function, xi, p, labels, bounds=None, print=True)

    assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"
    assert len(result) == len(labels), "DataFrame should have one row per variable"


def test_screening_plot_output(test_data):
    """Test if screening generates a plot when print=False."""
    X, labels, range_ = test_data
    xi = 0.1
    p = 3

    # Ensure no exceptions are raised during plotting
    try:
        screening(X, mock_objective_function, xi, p, labels, bounds=range_, print=False)
    except Exception as e:
        pytest.fail(f"Plotting failed with exception: {e}")