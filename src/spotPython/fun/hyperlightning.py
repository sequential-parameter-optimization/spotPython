import logging
import numpy as np
from numpy.random import default_rng
from numpy import array

# here we use train_model from spotPython.light.trainmodel
# and not from spot.light.traintest:
from spotPython.light.trainmodel import train_model
from spotPython.hyperparameters.values import (
    assign_values,
    generate_one_config_from_var_dict,
)

logger = logging.getLogger(__name__)
py_handler = logging.FileHandler(f"{__name__}.log", mode="w")
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
py_handler.setFormatter(py_formatter)
logger.addHandler(py_handler)


class HyperLightning:
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
            >>> from spotPython.utils.init import fun_control_init
                from spotPython.utils.file import get_experiment_name, get_spot_tensorboard_path
                from spotPython.utils.device import getDevice
                from spotPython.light.cnn.googlenet import GoogleNet
                from spotPython.data.lightning_hyper_dict import LightningHyperDict
                from spotPython.hyperparameters.values import add_core_model_to_fun_control
                from spotPython.fun.hyperlightning import HyperLightning
                from spotPython.hyperparameters.values import get_default_hyperparameters_as_array
                MAX_TIME = 1
                INIT_SIZE = 3
                WORKERS = 8
                PREFIX="TEST"
                experiment_name = get_experiment_name(prefix=PREFIX)
                fun_control = fun_control_init(
                    spot_tensorboard_path=get_spot_tensorboard_path(experiment_name),
                    num_workers=WORKERS,
                    device=getDevice(),
                    _L_in=3,
                    _L_out=10,
                    TENSORBOARD_CLEAN=True)
                add_core_model_to_fun_control(core_model=GoogleNet,
                                            fun_control=fun_control,
                                            hyper_dict= LightningHyperDict)
                X_start = get_default_hyperparameters_as_array(fun_control)
                hyper_light = HyperLightning(seed=126, log_level=50)
                hyper_light.fun(X=X_start, fun_control=fun_control)

        """
        z_res = np.array([], dtype=float)
        if fun_control is not None:
            self.fun_control.update(fun_control)
        self.check_X_shape(X)
        var_dict = assign_values(X, self.fun_control["var_name"])
        # type information and transformations are considered in generate_one_config_from_var_dict:
        for config in generate_one_config_from_var_dict(var_dict, self.fun_control):
            logger.debug(f"\nconfig: {config}")
            print(f"\ncore_model: {fun_control['core_model']}")
            print(f"config: {config}")
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
