import logging
import numpy as np
import pprint
from numpy.random import default_rng
from spotpython.light.trainmodel import train_model
from spotpython.hyperparameters.values import assign_values, generate_one_config_from_var_dict, get_var_name

logger = logging.getLogger(__name__)
py_handler = logging.FileHandler(f"{__name__}.log", mode="w")
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
py_handler.setFormatter(py_formatter)
logger.addHandler(py_handler)


class MoHyperLight:
    """
    Multi-Objective Hyperparameter Tuning for Lightning.

    Args:
        seed (int): seed for the random number generator. See Numpy Random Sampling.
        log_level (int): log level for the logger.

    Attributes:
        seed (int): seed for the random number generator.
        rng (Generator): random number generator.
        fun_control (dict): dictionary containing control parameters for the hyperparameter tuning.
        log_level (int): log level for the logger.

    Examples:
        >>> hyper_light = MoHyperLight(seed=126, log_level=50)
        >>> print(hyper_light.seed)
            126
    """

    def __init__(self, seed: int = 126, log_level: int = 50) -> None:
        self.seed = seed
        self.rng = default_rng(seed=self.seed)
        self.log_level = log_level
        logger.setLevel(log_level)
        logger.info(f"Starting the logger at level {log_level} for module {__name__}:")

    def check_X_shape(self, X: np.ndarray, fun_control: dict) -> np.ndarray:
        """
        Checks the shape of the input array X and raises an exception if it is not valid.

        Args:
            X (np.ndarray):
                input array.
            fun_control (dict):
                dictionary containing control parameters for the hyperparameter tuning.

        Returns:
            np.ndarray:
                input array with valid shape.

        Raises:
            Exception:
                if the shape of the input array is not valid.

        Examples:
            >>> import numpy as np
                from spotpython.utils.init import fun_control_init
                from spotpython.light.regression.netlightregression import NetLightRegression
                from spotpython.hyperdict.light_hyper_dict import LightHyperDict
                from spotpython.hyperparameters.values import add_core_model_to_fun_control
                from spotpython.fun.mohyperlight import MoHyperLight
                from spotpython.hyperparameters.values import get_var_name
                fun_control = fun_control_init()
                add_core_model_to_fun_control(core_model=NetLightRegression,
                                            fun_control=fun_control,
                                            hyper_dict=LightHyperDict)
                hyper_light = MoHyperLight(seed=126, log_level=50)
                n_hyperparams = len(get_var_name(fun_control))
                # generate a random np.array X with shape (2, n_hyperparams)
                X = np.random.rand(2, n_hyperparams)
                X == hyper_light.check_X_shape(X, fun_control)
                array([[ True,  True,  True,  True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True,  True,  True,  True]])

        """
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
        if X.shape[1] != len(get_var_name(fun_control)):
            raise Exception("Invalid shape of input array X.")
        return X

    def fun(self, X: np.ndarray, fun_control: dict = None) -> np.ndarray:
        """
        Evaluates the function for the given input array X of shape (n, k)
        and control parameters specified as a dict.
        Calls the train_model function from spotpython.light.trainmodel
        to train the model and evaluate the results.

        Args:
            X (np.ndarray):
                input array of shape (n, k) where n is the number of configurations evaluated
                and k is the number of hyperparameters.
            fun_control (dict):
                dictionary containing control parameters for the hyperparameter tuning.

        Returns:
            (np.ndarray):
                (2, n) array where the first row contains the evaluation results (z_res)
                and the second row contains the extracted "epochs" values (epochs_res).
        """
        z_res = np.array([], dtype=float)
        epochs_res = np.array([], dtype=float)  # Array to store "epochs" values

        self.check_X_shape(X=X, fun_control=fun_control)
        var_dict = assign_values(X, get_var_name(fun_control))

        # Type information and transformations are considered in generate_one_config_from_var_dict:
        for config in generate_one_config_from_var_dict(var_dict, fun_control):
            if fun_control["show_config"]:
                print("\nIn fun(): config:")
                pprint.pprint(config)
            logger.debug(f"\nconfig: {config}")

            try:
                logger.debug("fun: Calling train_model")
                df_eval = train_model(config, fun_control)
                logger.debug("fun: train_model returned")
            except Exception as err:
                if fun_control["verbosity"] > 0:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    if fun_control["verbosity"] > 1:
                        pprint.pprint(fun_control)
                    print(f"Error in fun(). Call to train_model failed. {err=}, {type(err)=}")
                    print("Setting df_eval to np.nan\n")
                logger.error(f"Error in fun(). Call to train_model failed. {err=}, {type(err)=}")
                logger.error("Setting df_eval to np.nan")
                df_eval = np.nan

            # Multiply results by the weights. Positive weights mean that the result is to be minimized.
            # Negative weights mean that the result is to be maximized, e.g., accuracy.
            z_val = fun_control["weights"] * df_eval
            z_res = np.append(z_res, z_val)  # Append evaluation result

            # Extract "epochs" from the config and append to epochs_res
            epochs_val = config.get("epochs", np.nan)  # Default to np.nan if "epochs" is not in config
            epochs_res = np.append(epochs_res, epochs_val)

        # Stack z_res and epochs_res into a (n, 2) array
        result = np.column_stack((z_res, epochs_res))
        print(f"result.shape: {result.shape}")
        print(f"result: {result}")

        return result
