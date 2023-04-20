from numpy.random import default_rng
import numpy as np
from numpy import array


from spotPython.hyperparameters.values import (
    assign_values,
)
from spotPython.hyperparameters.prepare import (
    get_dict_with_levels_and_types,
    get_one_config_from_var_dict,
    iterate_dict_values,
    convert_keys,
)

from spotPython.utils.transform import transform_hyper_parameter_values

import logging
import statistics
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
            "metric": metrics.MAE(),
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

    def compute_y(self, df_eval):
        """Compute the objective function value.
        Args:
            df_eval (pd.DataFrame): DataFrame with the evaluation results.
        Returns:
            (float): objective function value. Mean of the MAEs of the predicted values.
        Examples:
            >>> df_eval = pd.DataFrame( [[1, 2, 3], [4, 5, 6]], columns=['Metric', 'CompTime (s)', 'Memory (MB)'])
            >>> weights = [1, 1, 1]
            >>> compute_y(df_eval, weights)
            4.0
        """
        # take the mean of the MAEs/ACCs of the predicted values and ignore the NaN values
        df_eval = df_eval.dropna()
        y_error = df_eval["Metric"].mean()
        logger.debug("y_error from eval_oml_horizon: %s", y_error)
        y_r_time = df_eval["CompTime (s)"].mean()
        logger.debug("y_r_time from eval_oml_horizon: %s", y_r_time)
        y_memory = df_eval["Memory (MB)"].mean()
        logger.debug("y_memory from eval_oml_horizon: %s", y_memory)
        weights = self.fun_control["weights"]
        logger.debug("weights from eval_oml_horizon: %s", weights)
        y = weights[0] * y_error + weights[1] * y_r_time + weights[2] * y_memory
        logger.debug("weighted res from eval_oml_horizon: %s", y)
        return y

    
    def check_weights(self):
        if len(self.fun_control["weights"]) != 3:
            raise ValueError("The weights array must be of length 3.")

    def check_X_shape(self, X):
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
        if X.shape[1] != len(self.fun_control["var_name"]):
            raise Exception

    def evaluate_model(self, model, fun_control):
        try:
            df_eval, df_preds = eval_oml_horizon(
                model=model,
                train=fun_control["train"],
                test=fun_control["test"],
                target_column=fun_control["target_column"],
                horizon=fun_control["horizon"],
                oml_grace_period=fun_control["oml_grace_period"],
                metric=fun_control["metric_sklearn"],
            )
        except Exception as err:
            print(f"Error in fun_oml_horizon(). Call to eval_oml_horizon failed. {err=}, {type(err)=}")
        return df_eval, df_preds

    def fun_sklearn(self, X, fun_control=None, return_model=False, return_df=False):
        z_res = np.array([], dtype=float)
        self.fun_control.update(fun_control)
        self.check_weights()
        self.check_X_shape(X)
        var_dict = assign_values(X, self.fun_control["var_name"])
        for config in get_one_config_from_var_dict(var_dict, self.fun_control):
            model = compose.Pipeline(self.fun_control["prep_model"], self.fun_control["core_model"](**config))
            if return_model:
                return model
            df_eval, df_preds = self.evaluate_model(model, self.fun_control)
            if return_df:
                return df_eval, df_preds
            try:
                y = self.compute_y(df_eval)
            except Exception as err:
                y = np.nan
                print(f"Error in fun(). Call to evaluate failed. {err=}, {type(err)=}")
                print(f"Setting y to {y:.2f}.")
            z_res = np.append(z_res, y / self.fun_control["n_samples"])
        return z_res
