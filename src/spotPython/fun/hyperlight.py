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
                from spotpython.fun.hyperlight import HyperLight
                from spotpython.hyperparameters.values import get_var_name
                fun_control = fun_control_init()
                add_core_model_to_fun_control(core_model=NetLightRegression,
                                            fun_control=fun_control,
                                            hyper_dict=LightHyperDict)
                hyper_light = HyperLight(seed=126, log_level=50)
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
            >>> from spotpython.utils.init import fun_control_init
                from spotpython.light.regression.netlightregression import NetLightRegression
                from spotpython.hyperdict.light_hyper_dict import LightHyperDict
                from spotpython.hyperparameters.values import (add_core_model_to_fun_control,
                    get_default_hyperparameters_as_array)
                from spotpython.fun.hyperlight import HyperLight
                from spotpython.data.diabetes import Diabetes
                from spotpython.hyperparameters.values import set_control_key_value
                import numpy as np
                fun_control = fun_control_init(
                    _L_in=10,
                    _L_out=1,)
                dataset = Diabetes()
                set_control_key_value(control_dict=fun_control,
                          key="data_set",
                          value=dataset)
                add_core_model_to_fun_control(core_model=NetLightRegression,
                                            fun_control=fun_control,
                                            hyper_dict=LightHyperDict)
                hyper_light = HyperLight(seed=126, log_level=50)
                X = get_default_hyperparameters_as_array(fun_control)
                # combine X and X to a np.array with shape (2, n_hyperparams)
                # so that two values are returned
                X = np.vstack((X, X))
                hyper_light.fun(X, fun_control)
                array([27462.84179688, 20990.08007812])
        """
        z_res = np.array([], dtype=float)
        self.check_X_shape(X=X, fun_control=fun_control)
        var_dict = assign_values(X, get_var_name(fun_control))
        # type information and transformations are considered in generate_one_config_from_var_dict:
        for config in generate_one_config_from_var_dict(var_dict, fun_control):
            if fun_control["verbosity"] > 0:
                print("\nIn fun(): config:")
                pprint.pprint(config)
            logger.debug(f"\nconfig: {config}")
            # extract parameters like epochs, batch_size, lr, etc. from config
            # config_id = generate_config_id(config)
            try:
                logger.debug("fun: Calling train_model")
                df_eval = train_model(config, fun_control)
                logger.debug("fun: train_model returned")
            except Exception as err:
                if fun_control["verbosity"] > 0:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    pprint.pprint(fun_control)
                    print(f"Error in fun(). Call to train_model failed. {err=}, {type(err)=}")
                    print("Setting df_eval to np.nan\n")
                logger.error(f"Error in fun(). Call to train_model failed. {err=}, {type(err)=}")
                logger.error("Setting df_eval to np.nan")
                df_eval = np.nan
            z_val = fun_control["weights"] * df_eval
            z_res = np.append(z_res, z_val)
        return z_res
