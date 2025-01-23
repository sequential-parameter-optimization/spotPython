import pytest
from unittest.mock import MagicMock, create_autospec
import numpy as np
from spotpython.spot import Spot

@pytest.fixture
def mock_spot():
    # Create a mock Spot instance
    mock_spot = create_autospec(Spot)

    # Mock attributes
    mock_spot.X = np.array([[0], [1]])
    mock_spot.y = np.array([0, 1])
    mock_spot.var_name = ['x1']
    mock_spot.k = 1

    # Mock spot_writer to be a MagicMock
    mock_spot.spot_writer = MagicMock()

    return mock_spot

def test_write_tensorboard_log(mock_spot):
    # Call the method
    Spot.write_initial_tensorboard_log(mock_spot)

    # Define expected calls
    expected_calls = [
        ({"x1": 0}, {"hp_metric": 0}),
        ({"x1": 1}, {"hp_metric": 1})
    ]

    # Check if add_hparams was called for each expected configuration
    for config, metrics in expected_calls:
        mock_spot.spot_writer.add_hparams.assert_any_call(config, metrics)
    
    # Ensure that the flush method was called the expected number of times
    assert mock_spot.spot_writer.flush.call_count == len(mock_spot.y)