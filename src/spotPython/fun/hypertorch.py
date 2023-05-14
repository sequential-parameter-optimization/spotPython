from numpy.random import default_rng
import numpy as np
from numpy import array
from sklearn.pipeline import make_pipeline
from spotPython.torch.traintest import evaluate_cv, evaluate_hold_out


from spotPython.hyperparameters.values import (
    assign_values,
    generate_one_config_from_var_dict,
)

import logging

logger = logging.getLogger(__name__)
py_handler = logging.FileHandler(f"{__name__}.log", mode="w")
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
py_handler.setFormatter(py_formatter)
logger.addHandler(py_handler)


class HyperTorch:
    """
    Hyperparameter Tuning for Torch.

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
            "metric": None,
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

    def fun_torch(self, X, fun_control=None):
        z_res = np.array([], dtype=float)
        self.fun_control.update(fun_control)
        # print(self.fun_control)
        self.check_X_shape(X)
        var_dict = assign_values(X, self.fun_control["var_name"])
        # type information and transformations are considered in generate_one_config_from_var_dict:
        for config in generate_one_config_from_var_dict(var_dict, self.fun_control):
            if self.fun_control["prep_model"] is not None:
                model = make_pipeline(self.fun_control["prep_model"], self.fun_control["core_model"](**config))
            else:
                model = self.fun_control["core_model"](**config)
            try:
                if self.fun_control["eval"] == "train_cv":
                    df_eval, _ = evaluate_cv(
                        model,
                        dataset=fun_control["train"],
                        shuffle=self.fun_control["shuffle"],
                        device=self.fun_control["device"],
                        show_batch_interval=self.fun_control["show_batch_interval"],
                    )
                elif self.fun_control["eval"] == "test_cv":
                    df_eval, _ = evaluate_cv(
                        model,
                        dataset=fun_control["test"],
                        shuffle=self.fun_control["shuffle"],
                        device=self.fun_control["device"],
                        show_batch_interval=self.fun_control["show_batch_interval"],
                    )
                elif self.fun_control["eval"] == "test_hold_out":
                    df_eval, _ = evaluate_hold_out(
                        model,
                        train_dataset=fun_control["train"],
                        shuffle=self.fun_control["shuffle"],
                        test_dataset=fun_control["test"],
                        device=self.fun_control["device"],
                        show_batch_interval=self.fun_control["show_batch_interval"],
                        path=self.fun_control["path"],
                        save_model=self.fun_control["save_model"],
                    )
                else:  # eval == "train_hold_out"
                    df_eval, _ = evaluate_hold_out(
                        model,
                        train_dataset=fun_control["train"],
                        shuffle=self.fun_control["shuffle"],
                        device=self.fun_control["device"],
                        show_batch_interval=self.fun_control["show_batch_interval"],
                        path=self.fun_control["path"],
                        save_model=self.fun_control["save_model"],
                    )
            except Exception as err:
                print(f"Error in fun_torch(). Call to evaluate_model failed. {err=}, {type(err)=}")
                print("Setting df_eval to np.nan")
                df_eval = np.nan
            z_res = np.append(z_res, fun_control["weights"] * df_eval)
        return z_res
