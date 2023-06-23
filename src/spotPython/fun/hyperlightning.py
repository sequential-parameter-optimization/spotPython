import logging
from numpy.random import default_rng
from numpy import array
from spotPython.lightning.traintest import train_model
from spotPython.hyperparameters.values import (
    assign_values,
    generate_one_config_from_var_dict,
)
from spotPython.utils.eda import generate_config_id
import numpy as np


logger = logging.getLogger(__name__)
py_handler = logging.FileHandler(f"{__name__}.log", mode="w")
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
py_handler.setFormatter(py_formatter)
logger.addHandler(py_handler)


class HyperLightning:
    """
    Hyperparameter Tuning for Lightning.

    Args:
        seed (int): seed.
            See [Numpy Random Sampling](https://numpy.org/doc/stable/reference/random/index.html#random-quick-start)

    """

    def __init__(self, seed=126, log_level=50):
        self.seed = seed
        self.rng = default_rng(seed=self.seed)
        self.fun_control = {
            "seed": None,
            "data": None,
            "step": 10_000,
            "horizon": None,
            "grace_period": None,
            "metric_river": None,
            "metric_sklearn": None,
            "weights": array([1, 0, 0]),
            "weight_coeff": 0.0,
            "log_level": log_level,
            "var_name": [],
            "var_type": [],
        }
        self.log_level = self.fun_control["log_level"]
        logger.setLevel(self.log_level)
        logger.info(f"Starting the logger at level {self.log_level} for module {__name__}:")

    def check_X_shape(self, X):
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
        if X.shape[1] != len(self.fun_control["var_name"]):
            raise Exception

    def fun_lightning(self, X, fun_control=None):
        z_res = np.array([], dtype=float)
        self.fun_control.update(fun_control)
        self.check_X_shape(X)
        var_dict = assign_values(X, self.fun_control["var_name"])
        # type information and transformations are considered in generate_one_config_from_var_dict:
        for config in generate_one_config_from_var_dict(var_dict, self.fun_control):
            print(f"\nconfig: {config}")
            # extract parameters like epochs, batch_size, lr, etc. from config
            config_id = generate_config_id(config)
            model = self.fun_control["core_model"](**config)
            try:
                df_eval, _ = train_model(
                    net=model,
                    model_name=config_id,
                    dataset=fun_control["train"],
                    shuffle=self.fun_control["shuffle"],
                    device=self.fun_control["device"],
                    # task=self.fun_control["task"],
                    # writer=self.fun_control["writer"],
                    # writerId=config_id,
                )
            except Exception as err:
                print(f"Error in fun_lightning(). Call to evaluate_model failed. {err=}, {type(err)=}")
                print("Setting df_eval to np.nan")
                df_eval = np.nan
            z_val = fun_control["weights"] * df_eval
            if self.fun_control["writer"] is not None:
                writer = self.fun_control["writer"]
                writer.add_hparams(config, {"fun_lightning: loss": z_val})
                writer.flush()
            z_res = np.append(z_res, z_val)
        return z_res
