import os
from unittest.mock import patch

from spotPython.utils.init import get_spot_tensorboard_path


def test_get_spot_tensorboard_path():
    experiment_name = "00_ubuntu_2021-08-31_14-30-00"
    expected_default_path = os.path.join("runs/spot_logs", experiment_name)
    custom_path = "/custom/path"

    # Test with default path
    with patch.dict(os.environ, {}, clear=True):
        assert get_spot_tensorboard_path(experiment_name) == expected_default_path

    # Test with custom environment path
    with patch.dict(os.environ, {"PATH_TENSORBOARD": custom_path}):
        assert get_spot_tensorboard_path(experiment_name) == os.path.join(custom_path, experiment_name)
