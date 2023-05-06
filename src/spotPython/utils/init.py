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
                'shuffle': None}
    """
    fun_control = {
        "data": None,
        "train": None,
        "test": None,
        "n_samples": None,
        "target_column": None,
        "shuffle": None,
        "eval": None,
    }
    return fun_control
