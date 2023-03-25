def modify_hyper_parameter_levels(fun_control, hyperparameter, levels):
    """

    Args:
        fun_control (dict): fun_control dictionary
        hyperparameter (str): hyperparameter name
        levels (list): list of levels

    Returns:
        fun_control (dict): updated fun_control
    Example:
        >>> fun_control = {}
            core_model  = HoeffdingTreeRegressor
            fun_control.update({"core_model": core_model})
            fun_control.update({"core_model_hyper_dict": river_hyper_dict[core_model.__name__]})
            levels = ["mean", "model"]
            fun_control = modify_hyper_parameter_levels(fun_control, "leaf_prediction", levels)
    """
    fun_control["core_model_hyper_dict"][hyperparameter].update({"levels": levels})
    fun_control["core_model_hyper_dict"][hyperparameter].update({"lower": 0})
    fun_control["core_model_hyper_dict"][hyperparameter].update({"upper": len(levels) - 1})
    return fun_control
