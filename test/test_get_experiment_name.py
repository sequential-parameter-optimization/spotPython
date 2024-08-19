from unittest.mock import patch
from spotpython.utils.init import get_experiment_name
import datetime


def test_get_experiment_name():
    mock_hostname = "ubuntu"
    # Create a datetime without timezone information
    mock_time = datetime.datetime(2021, 8, 31, 14, 30, 0)
    # Format the mock time accordingly
    expected_time_str = mock_time.strftime("%Y-%m-%d_%H-%M-%S")
    prefix = "00"

    with patch("socket.gethostname", return_value=mock_hostname):
        with patch("datetime.datetime") as mock_datetime:
            # Set the mock to return the custom mock time
            mock_datetime.now.return_value = mock_time
            mock_datetime.side_effect = lambda *args, kw: datetime.datetime(*args, kw)

            # Test with default prefix
            expected_experiment_name = f"{prefix}_{mock_hostname}_{expected_time_str}"
            assert get_experiment_name() == expected_experiment_name

            # Test with custom prefix
            custom_prefix = "01"
            expected_experiment_name_custom = f"{custom_prefix}_{mock_hostname}_{expected_time_str}"
            assert get_experiment_name(custom_prefix) == expected_experiment_name_custom
