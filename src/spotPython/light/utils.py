from spotPython.hyperparameters.values import get_one_config_from_X


def get_tuned_architecture(spot_tuner, fun_control):
    X = spot_tuner.to_all_dim(spot_tuner.min_X.reshape(1, -1))
    config = get_one_config_from_X(X, fun_control)
    return config
