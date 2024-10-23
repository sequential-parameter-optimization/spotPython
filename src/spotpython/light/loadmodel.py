from spotpython.utils.eda import generate_config_id
import os
from typing import Any


def load_light_from_checkpoint(config: dict, fun_control: dict, postfix: str = "_TEST") -> Any:
    """
    Loads a model from a checkpoint using the given configuration and function control parameters.

    Notes:
        * `load_light_from_checkpoint` loads the last checkpoint of the model
        * Randomness, dropout, etc... are disabled.

    References:
        * https://pytorch-lightning.readthedocs.io/en/0.8.5/weights_loading.html

    Args:
        config (dict):
            A dictionary containing the configuration parameters for the model.
        fun_control (dict):
            A dictionary containing the function control parameters.
        postfix (str):
            The postfix to append to the configuration ID when generating the checkpoint path.
            Default is "_TEST". Can be set to "_TRAIN" for training checkpoints.

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
        ...     "_torchmetric": "mean_squared_error",
        ...     "core_model": MyModel,
        ...     "TENSORBOARD_PATH": "./tensorboard",
        ... }
        >>> model = load_light_from_checkpoint(config, fun_control)
    """
    print(f"config: {config}")
    # load a model from a checkpoint with the same config_id
    # that was used in the test phase. Therefore, no timestamp is added.
    config_id = generate_config_id(config, timestamp=False) + postfix
    default_root_dir = os.path.join(fun_control["CHECKPOINT_PATH"], config_id, "last.ckpt")
    print(f"Loading model with {config_id} from {default_root_dir}")
    model = fun_control["core_model"].load_from_checkpoint(
        default_root_dir,
        _L_in=fun_control["_L_in"],
        _L_out=fun_control["_L_out"],
        _L_cond=fun_control["_L_cond"],
        _torchmetric=fun_control["_torchmetric"],
    )
    # disable randomness, dropout, etc...
    print(f"Model: {model}")
    model.eval()
    return model
