from numpy.random import default_rng
import numpy as np
from numpy import array
from sklearn.pipeline import make_pipeline
from spotPython.utils.convert import get_Xy_from_df


from spotPython.hyperparameters.values import assign_values
from spotPython.hyperparameters.prepare import (
    get_one_config_from_var_dict,
)

import logging
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)
# configure the handler and formatter as needed
py_handler = logging.FileHandler(f"{__name__}.log", mode="w")
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
# add formatter to the handler
py_handler.setFormatter(py_formatter)
# add handler to the logger
logger.addHandler(py_handler)


class HyperSklearn:
    """
    Hyperparameter Tuning for Sklearn.

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
            "metric_sklearn": mean_absolute_error,
            "weights": array([1, 0, 0]),
            "weight_coeff": 0.0,
            "log_level": log_level,
            "var_name": [],
            "var_type": [],
            "prep_model": None,
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

    def evaluate_model(self, model, fun_control):
        try:
            X_train, y_train = get_Xy_from_df(fun_control["train"], fun_control["target_column"])
            X_test, y_test = get_Xy_from_df(fun_control["test"], fun_control["target_column"])
            model.fit(X_train, y_train)
            df_preds = model.predict(X_test)
            df_eval = fun_control["metric_sklearn"](y_test, df_preds)
        except Exception as err:
            print(f"Error in fun_sklearn(). Call to evaluate_model failed. {err=}, {type(err)=}")
            df_eval = np.nan
            df_eval = np.nan
        return df_eval, df_preds

    def get_sklearn_df_eval_preds(self, model):
        try:
            df_eval, df_preds = self.evaluate_model(model, self.fun_control)
        except Exception as err:
            print(f"Error in get_sklearn_df_eval_preds(). Call to evaluate_model failed. {err=}, {type(err)=}")
            print("Setting df_eval and df.preds to np.nan")
            df_eval = np.nan
            df_preds = np.nan
        return df_eval, df_preds

    def fun_sklearn(self, X, fun_control=None):
        z_res = np.array([], dtype=float)
        self.fun_control.update(fun_control)
        self.check_X_shape(X)
        var_dict = assign_values(X, self.fun_control["var_name"])
        for config in get_one_config_from_var_dict(var_dict, self.fun_control):
            if self.fun_control["prep_model"] is not None:
                model = make_pipeline(self.fun_control["prep_model"], self.fun_control["core_model"](**config))
            else:
                model = self.fun_control["core_model"](**config)
            try:
                df_eval, _ = self.evaluate_model(model, self.fun_control)
            except Exception as err:
                print(f"Error in fun_sklearn(). Call to evaluate_model failed. {err=}, {type(err)=}")
                print("Setting df_eval to np.nan")
                df_eval = np.nan
            z_res = np.append(z_res, fun_control["weights"] * df_eval)
        return z_res
