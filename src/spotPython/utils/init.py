def fun_control_init():
    """Initialize fun_control dictionary.
    Args:
        None
        Returns:
            fun_control (dict): A dictionary containing the information about the core model and the hyperparameters.
        Example:
            >>> fun_control_init()
            {'data': None,
                'train': None,
                'test': None,
                'n_samples': None,
                'target_column': None,
                'shuffle': None,
                'k_folds': None,
                'device': None}
    """
    fun_control = {
        "criterion": None,
        "data": None,
        "train": None,
        "test": None,
        "n_samples": None,
        "target_column": None,
        "shuffle": None,
        "eval": None,
        "k_folds": None,
        "optimizer": None,
        "device": None,
        "metric_params": {},
        "show_batch_interval": 1_000_000,
        "path": None,
        "save_model": False,
    }
    return fun_control
