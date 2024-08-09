import pickle
import pytest
from unittest.mock import patch, MagicMock

from spotPython.utils.file import load_and_run_spot_python_experiment


@pytest.fixture
def mock_experiment_data():
    return (
        MagicMock(),  # mock spot_tuner
        {"TENSORBOARD_CLEAN": True, "tensorboard_start": False},  # mock fun_control
        MagicMock(),  # mock design_control
        MagicMock(),  # mock surrogate_control
        MagicMock()   # mock optimizer_control
    )


@patch("spotPython.utils.file.load_experiment")
@patch("spotPython.utils.file.pprint.pprint")
@patch("spotPython.utils.file.gen_design_table")
@patch("spotPython.utils.file.setup_paths")
@patch("spotPython.utils.file.start_tensorboard")
@patch("spotPython.utils.file.stop_tensorboard")
def test_load_and_run_spot_python_experiment(
        mock_stop_tensorboard,
        mock_start_tensorboard,
        mock_setup_paths,
        mock_gen_design_table,
        mock_pprint,
        mock_load_experiment,
        mock_experiment_data
    ):
    # Set up the mocks
    mock_load_experiment.return_value = mock_experiment_data
    mock_start_tensorboard.return_value = None

    # Call the function under test
    result = load_and_run_spot_python_experiment("spot_test_experiment.pkl")

    # Verify behaviors for load_experiment and return values
    mock_load_experiment.assert_called_once_with("spot_test_experiment.pkl")
    mock_gen_design_table.assert_called_once_with(mock_experiment_data[1])
    mock_setup_paths.assert_called_once_with(mock_experiment_data[1]["TENSORBOARD_CLEAN"])
    mock_experiment_data[0].init_spot_writer.assert_called_once()
    mock_experiment_data[0].run.assert_called_once()
    mock_experiment_data[0].save_experiment.assert_called_once()
    mock_stop_tensorboard.assert_called_once_with(None)

    # Assert the returned tuple
    assert result == (*mock_experiment_data, None)
