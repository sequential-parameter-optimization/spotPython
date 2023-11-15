from spotPython.hyperparameters.values import get_one_config_from_X
from typing import Any


def get_tuned_architecture(spot_tuner, fun_control) -> dict:
    """
    Returns the tuned architecture.

    Args:
        spot_tuner (object):
            spot tuner object.
        fun_control (dict):
            dictionary containing control parameters for the hyperparameter tuning.

    Returns:
        (dict):
            dictionary containing the tuned architecture.
    """
    X = spot_tuner.to_all_dim(spot_tuner.min_X.reshape(1, -1))
    config = get_one_config_from_X(X, fun_control)
    return config


def create_model(config, fun_control, **kwargs) -> object:
    """
    Creates a model for the given configuration and control parameters.

    Args:
        config (dict):
            dictionary containing the configuration for the hyperparameter tuning.
        fun_control (dict):
            dictionary containing control parameters for the hyperparameter tuning.
        **kwargs (Any):
            additional keyword arguments.

    Returns:
        (object):
            model object.
    """
    return fun_control["core_model"](**config, **kwargs)
