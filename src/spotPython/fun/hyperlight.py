import logging
import numpy as np
from numpy.random import default_rng
from numpy import array
from spotPython.light.traintest import train_model
from spotPython.hyperparameters.values import (
    assign_values,
    generate_one_config_from_var_dict,
)

logger = logging.getLogger(__name__)
py_handler = logging.FileHandler(f"{__name__}.log", mode="w")
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
py_handler.setFormatter(py_formatter)
logger.addHandler(py_handler)


class HyperLight:
    """
    Hyperparameter Tuning for Lightning.

    Args:
        seed (int): seed for the random number generator. See Numpy Random Sampling.
        log_level (int): log level for the logger.

    Attributes:
        seed (int): seed for the random number generator.
        rng (Generator): random number generator.
        fun_control (dict): dictionary containing control parameters for the hyperparameter tuning.
        log_level (int): log level for the logger.

    Examples:
        >>> hyper_light = HyperLight(seed=126, log_level=50)
        >>> print(hyper_light.seed)
        126
    """

    def __init__(self, seed: int = 126, log_level: int = 50) -> None:
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

    def check_X_shape(self, X: np.ndarray) -> np.ndarray:
        """
        Checks the shape of the input array X and raises an exception if it is not valid.

        Args:
            X (np.ndarray):
                input array.

        Returns:
            np.ndarray:
                input array with valid shape.

        Raises:
            Exception:
                if the shape of the input array is not valid.

        Examples:
            >>> hyper_light = HyperLight(seed=126, log_level=50)
            >>> X = np.array([[1, 2], [3, 4]])
            >>> hyper_light.check_X_shape(X)
            array([[1, 2],
                   [3, 4]])
        """
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
        if X.shape[1] != len(self.fun_control["var_name"]):
            raise Exception("Invalid shape of input array X.")
        return X

    def fun(self, X: np.ndarray, fun_control: dict = None) -> np.ndarray:
        """
        Evaluates the function for the given input array X and control parameters.

        Args:
            X (np.ndarray):
                input array.
            fun_control (dict):
                dictionary containing control parameters for the hyperparameter tuning.

        Returns:
            (np.ndarray):
                array containing the evaluation results.

        Examples:
            >>> hyper_light = HyperLight(seed=126, log_level=50)
                X = np.array([[1, 2], [3, 4]])
                fun_control = {"weights": np.array([1, 0, 0])}
                hyper_light.fun(X, fun_control)
                array([nan, nan])
        """
        z_res = np.array([], dtype=float)
        if fun_control is not None:
            self.fun_control.update(fun_control)
        self.check_X_shape(X)
        var_dict = assign_values(X, self.fun_control["var_name"])
        # type information and transformations are considered in generate_one_config_from_var_dict:
        for config in generate_one_config_from_var_dict(var_dict, self.fun_control):
            logger.debug(f"\nconfig: {config}")
            # extract parameters like epochs, batch_size, lr, etc. from config
            # config_id = generate_config_id(config)
            try:
                print("fun: Calling train_model")
                df_eval = train_model(config, self.fun_control)
                print("fun: train_model returned")
            except Exception as err:
                logger.error(f"Error in fun(). Call to train_model failed. {err=}, {type(err)=}")
                logger.error("Setting df_eval to np.nan")
                df_eval = np.nan
            z_val = self.fun_control["weights"] * df_eval
            z_res = np.append(z_res, z_val)
        return z_res
