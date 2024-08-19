from spotpython.utils.file import get_experiment_from_PREFIX
import pytest
from unittest.mock import patch


def test_get_experiment_from_PREFIX_invalid_prefix():
    PREFIX = "invalid"

    with patch(
        "spotpython.utils.file.get_experiment_filename", return_value=None
    ) as mock_get_experiment_filename, patch(
        "spotpython.utils.file.load_experiment", side_effect=FileNotFoundError("Experiment not found")
    ) as mock_load_experiment:
        with pytest.raises(FileNotFoundError, match="Experiment not found"):
            get_experiment_from_PREFIX(PREFIX)

        # Ensure the filename function was called
        mock_get_experiment_filename.assert_called_once_with(PREFIX)
        # Ensure the load experiment function was called
        mock_load_experiment.assert_called_once()
