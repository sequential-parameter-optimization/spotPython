import logging
from numpy.random import default_rng
import numpy as np
from numpy import array
from sklearn.pipeline import make_pipeline
from spotpython.torch.traintest import evaluate_cv, evaluate_hold_out
from spotpython.hyperparameters.values import (
    assign_values,
    generate_one_config_from_var_dict,
)
from spotpython.utils.eda import generate_config_id

logger = logging.getLogger(__name__)
py_handler = logging.FileHandler(f"{__name__}.log", mode="w")
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
py_handler.setFormatter(py_formatter)
logger.addHandler(py_handler)


class HyperTorch:
    """
    Hyperparameter Tuning for Torch.

    Args:
        seed (int): seed for random number generator.
            See Numpy Random Sampling
        log_level (int): log level for logger. Default is 50.

    Attributes:
        seed (int): seed for random number generator.
        rng (Generator): random number generator.
        fun_control (dict): dictionary containing control parameters for the function.
        log_level (int): log level for logger.
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

    def check_X_shape(self, X: np.ndarray) -> None:
        """
        Check the shape of the input array X.

        Args:
            X (np.ndarray): input array.

        Raises:
            Exception: if the second dimension of X does not match the length of var_name in fun_control.

        Examples:
            >>> from spotpython.fun.hypertorch import HyperTorch
            >>> import numpy as np
            >>> hyper_torch = HyperTorch(seed=126, log_level=50)
            >>> hyper_torch.fun_control["var_name"] = ["x1", "x2"]
            >>> hyper_torch.check_X_shape(np.array([[1, 2], [3, 4]]))
            >>> hyper_torch.check_X_shape(np.array([1, 2]))
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

    def fun_torch(self, X: np.ndarray, fun_control: dict = None) -> np.ndarray:
        """
        Function to be optimized.

        Args:
            X (np.ndarray): input array.
            fun_control (dict): dictionary containing control parameters for the function.
        Returns:
            np.ndarray: output array.

        Examples:
            >>> from spotpython.fun.hypertorch import HyperTorch
            >>> import numpy as np
            >>> hyper_torch = HyperTorch(seed=126, log_level=50)
            >>> hyper_torch.fun_control["var_name"] = ["x1", "x2"]
            >>> hyper_torch.fun_torch(np.array([[1, 2], [3, 4]]))

        """
        z_res = np.array([], dtype=float)
        self.fun_control.update(fun_control)
        self.check_X_shape(X)
        var_dict = assign_values(X, self.fun_control["var_name"])
        for config in generate_one_config_from_var_dict(var_dict, self.fun_control):
            print(f"\nconfig: {config}")
            config_id = generate_config_id(config)
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
                        task=self.fun_control["task"],
                        writer=self.fun_control["spot_writer"],
                        writerId=config_id,
                    )
                elif self.fun_control["eval"] == "test_cv":
                    df_eval, _ = evaluate_cv(
                        model,
                        dataset=fun_control["test"],
                        shuffle=self.fun_control["shuffle"],
                        device=self.fun_control["device"],
                        show_batch_interval=self.fun_control["show_batch_interval"],
                        task=self.fun_control["task"],
                        writer=self.fun_control["spot_writer"],
                        writerId=config_id,
                    )
                elif self.fun_control["eval"] == "test_hold_out":
                    df_eval, _ = evaluate_hold_out(
                        model,
                        train_dataset=fun_control["train"],
                        shuffle=self.fun_control["shuffle"],
                        loss_function=self.fun_control["loss_function"],
                        metric=self.fun_control["metric_torch"],
                        test_dataset=fun_control["test"],
                        device=self.fun_control["device"],
                        show_batch_interval=self.fun_control["show_batch_interval"],
                        path=self.fun_control["path"],
                        task=self.fun_control["task"],
                        writer=self.fun_control["spot_writer"],
                        writerId=config_id,
                    )
                else:  # eval == "train_hold_out"
                    df_eval, _ = evaluate_hold_out(
                        model,
                        train_dataset=fun_control["train"],
                        shuffle=self.fun_control["shuffle"],
                        loss_function=self.fun_control["loss_function"],
                        metric=self.fun_control["metric_torch"],
                        device=self.fun_control["device"],
                        show_batch_interval=self.fun_control["show_batch_interval"],
                        path=self.fun_control["path"],
                        task=self.fun_control["task"],
                        writer=self.fun_control["spot_writer"],
                        writerId=config_id,
                    )
            except Exception as err:
                print(f"Error in fun_torch(). Call to evaluate_model failed. {err=}, {type(err)=}")
                print("Setting df_eval to np.nan")
                df_eval = np.nan
            z_val = fun_control["weights"] * df_eval
            if self.fun_control["spot_writer"] is not None:
                writer = self.fun_control["spot_writer"]
                writer.add_hparams(config, {"fun_torch: loss": z_val})
                writer.flush()
            z_res = np.append(z_res, z_val)
        return z_res
