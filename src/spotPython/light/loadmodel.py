from spotPython.utils.eda import generate_config_id
import os
from typing import Any


def load_light_from_checkpoint(config: dict, fun_control: dict, postfix: str = "_TEST") -> Any:
    """
    Loads a model from a checkpoint using the given configuration and function control parameters.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model.
        fun_control (dict): A dictionary containing the function control parameters.
        postfix (str): The postfix to append to the configuration ID when generating the checkpoint path.

    Returns:
        Any: The loaded model.

    Examples:
        >>> config = {
        ...     "initialization": "Xavier",
        ...     "batch_size": 32,
        ...     "patience": 10,
        ... }
        >>> fun_control = {
        ...     "_L_in": 10,
        ...     "_L_out": 1,
        ...     "core_model": MyModel,
        ...     "TENSORBOARD_PATH": "./tensorboard",
        ... }
        >>> model = load_light_from_checkpoint(config, fun_control)
    """
    config_id = generate_config_id(config) + postfix
    # default_root_dir = fun_control["TENSORBOARD_PATH"] + "lightning_logs/" + config_id + "/checkpoints/last.ckpt"
    default_root_dir = os.path.join(fun_control["CHECKPOINT_PATH"], config_id, "last.ckpt")
    # default_root_dir = os.path.join(fun_control["CHECKPOINT_PATH"], config_id)
    print(f"Loading model from {default_root_dir}")
    model = fun_control["core_model"].load_from_checkpoint(
        default_root_dir, _L_in=fun_control["_L_in"], _L_out=fun_control["_L_out"]
    )
    # disable randomness, dropout, etc...
    model.eval()
    return model
