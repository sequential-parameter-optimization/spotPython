from spotpython.utils.tensorboard import start_tensorboard, stop_tensorboard

import subprocess
from unittest.mock import patch, MagicMock
import pytest


@patch("subprocess.Popen")
def test_start_tensorboard(mock_popen):
    # Arrange
    mock_process = MagicMock()
    mock_popen.return_value = mock_process

    # Act
    process = start_tensorboard()

    # Assert
    mock_popen.assert_called_once_with(["tensorboard", "--logdir=./runs"])
    assert process == mock_process


@patch("subprocess.Popen")
def test_stop_tensorboard_active_process(mock_popen):
    # Arrange
    mock_process = MagicMock()
    mock_process.poll.return_value = None  # Simulate that the process is still running

    # Act
    stop_tensorboard(mock_process)

    # Assert
    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once()


@patch("subprocess.Popen")
def test_stop_tensorboard_no_process(mock_popen):
    # Arrange
    mock_process = MagicMock()
    mock_process.poll.return_value = 0  # Simulate that the process is already terminated

    # Act
    stop_tensorboard(mock_process)

    # Assert
    mock_process.terminate.assert_not_called()
    mock_process.wait.assert_not_called()


@patch("subprocess.Popen")
def test_stop_tensorboard_no_process_provided(mock_popen):
    # Act
    stop_tensorboard(None)

    # Assert
    # Nothing to assert because the function should just print a message, but ensuring no errors
    assert True