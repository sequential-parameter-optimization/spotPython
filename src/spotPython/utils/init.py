import os

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter


def fun_control_init(task, tensorboard_path=None, num_workers=0, device=None):
    """Initialize fun_control dictionary.
    Args:
        None
    Returns:
    fun_control (dict): A dictionary containing the information about the core model, loss function, metrics,
    and the hyperparameters.
    Example:
        >>> fun_control = fun_control_init("regression")
        >>> fun_control
        {'data': None,
        'train': None,
        'test': None,
        'loss_function': None,
        'metric_sklearn': None,
        'metric_river': None,
        'metric_torch': None,
        'metric_params': {},
        'prep_model': None,
        'n_samples': None,
        'target_column': None,
        'shuffle': None,
        'eval': None,
        'k_folds': None,
        'optimizer': None,
        'device': None,
        'show_batch_interval': 1000000,
        'path': None,
        'task': "regression",
        'save_model': False}
    """
    if task not in ["regression", "classification"]:
        raise Exception("task must be either 'regression' or 'classification'")
    if tensorboard_path is not None:
        if not isinstance(tensorboard_path, str):
            raise Exception("tensorboard_path must be a string")
        # create tensorboard_path if it does not exist
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
        # Starting with v0.2.41, Summary Writer is not initialized here but by Lightning
        # writer = SummaryWriter(tensorboard_path)
    # else:
    #     writer = None

    fun_control = {
        "data": None,
        "data_dir": "./data",
        "train": None,
        "test": None,
        "loss_function": None,
        "metric_sklearn": None,
        "metric_river": None,
        "metric_torch": None,
        "metric_params": {},
        "num_workers": num_workers,
        "prep_model": None,
        "n_samples": None,
        "target_column": None,
        "shuffle": None,
        "eval": None,
        "k_folds": None,
        "optimizer": None,
        "device": device,
        "show_batch_interval": 1_000_000,
        "path": None,
        "task": task,
        "tensorboard_path": tensorboard_path,
        "save_model": False,
        "weights": 1.0,
    }
    return fun_control
