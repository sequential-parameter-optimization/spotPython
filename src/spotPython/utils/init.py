def fun_control_init():
    """Initialize fun_control dictionary.
    Args:
        None
    Returns:
    fun_control (dict): A dictionary containing the information about the core model, loss function, metrics,
    and the hyperparameters.
    Example:
        >>> fun_control = fun_control_init()
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
        'save_model': False}
    """
    fun_control = {
        "data": None,
        "train": None,
        "test": None,
        "loss_function": None,
        "metric_sklearn": None,
        "metric_river": None,
        "metric_torch": None,
        "metric_params": {},
        "prep_model": None,
        "n_samples": None,
        "target_column": None,
        "shuffle": None,
        "eval": None,
        "k_folds": None,
        "optimizer": None,
        "device": None,
        "show_batch_interval": 1_000_000,
        "path": None,
        "save_model": False,
        "weights": 1.0,
    }
    return fun_control
