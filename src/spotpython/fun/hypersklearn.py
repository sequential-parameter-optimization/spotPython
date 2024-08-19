from numpy.random import default_rng
import numpy as np
from numpy import array
from sklearn.pipeline import make_pipeline
from spotpython.hyperparameters.values import assign_values
from spotpython.hyperparameters.values import (
    generate_one_config_from_var_dict,
)
from spotpython.sklearn.traintest import evaluate_cv, evaluate_model_oob, evaluate_hold_out, evaluate_model
import logging
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)
py_handler = logging.FileHandler(f"{__name__}.log", mode="w")
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
py_handler.setFormatter(py_formatter)
logger.addHandler(py_handler)


class HyperSklearn:
    """
    Hyperparameter Tuning for Sklearn.

    Args:
        seed (int): seed.
            See Numpy Random Sampling
        log_level (int): log level for logger. Default is 50.

    Attributes:
        seed (int): seed for random number generator.
        rng (Generator): random number generator.
        fun_control (dict): dictionary containing control parameters for the function.
        log_level (int): log level for logger.

    Examples:
        >>> from spotpython.fun.hypersklearn import HyperSklearn
        >>> hyper_sklearn = HyperSklearn(seed=126, log_level=50)
        >>> print(hyper_sklearn.seed)
        126
    """

    def __init__(self, seed: int = 126, log_level: int = 50):
        self.seed = seed
        self.rng = default_rng(seed=self.seed)
        self.fun_control = {
            "seed": None,
            "data": None,
            "step": 10_000,
            "horizon": None,
            "grace_period": None,
            "metric_river": None,
            "metric_sklearn": mean_absolute_error,
            "weights": array([1, 0, 0]),
            "weight_coeff": 0.0,
            "log_level": log_level,
            "var_name": [],
            "var_type": [],
            "prep_model": None,
            "predict_proba": False,
        }
        self.log_level = self.fun_control["log_level"]
        logger.setLevel(self.log_level)
        logger.info(f"Starting the logger at level {self.log_level} for module {__name__}:")

    def check_X_shape(self, X: np.ndarray) -> None:
        """
        Check the shape of the input array X.

        Args:
            X (np.ndarray): input array.

        Raises:
            Exception: if the second dimension of X does not match the length of var_name in fun_control.

        Examples:
            >>> from spotpython.fun.hypersklearn import HyperSklearn
            >>> hyper_sklearn = HyperSklearn(seed=126, log_level=50)
            >>> hyper_sklearn.fun_control["var_name"] = ["a", "b", "c"]
            >>> hyper_sklearn.check_X_shape(X=np.array([[1, 2, 3]]))
            >>> hyper_sklearn.check_X_shape(X=np.array([[1, 2]]))
            Traceback (most recent call last):
            ...
            Exception

        """
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
        if X.shape[1] != len(self.fun_control["var_name"]):
            raise Exception

    def get_sklearn_df_eval_preds(self, model) -> tuple:
        """
        Get evaluation and prediction dataframes for a given model.
        Args:
            model (sklearn model): sklearn model.

        Returns:
            (tuple): tuple containing evaluation and prediction dataframes.

        Raises:
            Exception: if call to evaluate_model fails.

        """
        try:
            df_eval, df_preds = self.evaluate_model(model, self.fun_control)
        except Exception as err:
            print(f"Error in get_sklearn_df_eval_preds(). Call to evaluate_model failed. {err=}, {type(err)=}")
            print("Setting df_eval and df.preds to np.nan")
            df_eval = np.nan
            df_preds = np.nan
        return df_eval, df_preds

    def fun_sklearn(self, X: np.ndarray, fun_control: dict = None) -> np.ndarray:
        """
        Evaluate a sklearn model using hyperparameters specified in X.

        Args:
            X (np.ndarray): input array containing hyperparameters.
            fun_control (dict): dictionary containing control parameters for the function. Default is None.

        Returns:
            (np.ndarray): array containing evaluation results.

        Raises:
            Exception: if call to evaluate_model fails.

        """
        z_res = np.array([], dtype=float)
        self.fun_control.update(fun_control)
        self.check_X_shape(X)
        var_dict = assign_values(X, self.fun_control["var_name"])
        for config in generate_one_config_from_var_dict(var_dict, self.fun_control):
            if self.fun_control["prep_model"] is not None:
                model = make_pipeline(self.fun_control["prep_model"](), self.fun_control["core_model"](**config))
            else:
                model = self.fun_control["core_model"](**config)
            try:
                eval_type = fun_control["eval"]
                if eval_type == "eval_test":
                    df_eval, _ = evaluate_model(model, self.fun_control)
                elif eval_type == "eval_oob_score":
                    df_eval, _ = evaluate_model_oob(model, self.fun_control)
                elif eval_type == "train_cv":
                    df_eval, _ = evaluate_cv(model, self.fun_control)
                else:  # None or "evaluate_hold_out":
                    df_eval, _ = evaluate_hold_out(model, self.fun_control)
            except Exception as err:
                print(f"Error in fun_sklearn(). Call to evaluate_model failed. {err=}, {type(err)=}")
                print("Setting df_eval to np.nan")
                df_eval = np.nan
            z_res = np.append(z_res, fun_control["weights"] * df_eval)
        return z_res
