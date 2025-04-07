from __future__ import annotations

from torch.utils.tensorboard import SummaryWriter
import pickle
import pprint
import os
import copy
import json
from numpy.random import default_rng
from spotpython.design.spacefilling import SpaceFilling

# old Kriging with attribute "name" kriging
from spotpython.build.kriging import Kriging as OldKriging

# new Kriging without attribute "name" Kriging
from spotpython.surrogate.kriging import Kriging
from spotpython.utils.repair import apply_penalty_NA
from spotpython.utils.seed import set_all_seeds
import numpy as np
import pandas as pd
import pylab
from scipy import optimize
from math import isfinite

import matplotlib
import matplotlib.pyplot as plt

from numpy import argmin
from numpy import repeat
from numpy import sqrt
from numpy import spacing
from numpy import append
from numpy import min, max
from spotpython.utils.convert import get_shape
from spotpython.utils.init import fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
from spotpython.utils.compare import selectNew
from spotpython.utils.aggregate import aggregate_mean_var, select_distant_points
from spotpython.utils.repair import remove_nan, repair_non_numeric
from spotpython.utils.file import get_experiment_filename, get_result_filename
from spotpython.budget.ocba import get_ocba_X
import logging
import time
from spotpython.utils.progress import progress_bar
from spotpython.utils.convert import find_indices, sort_by_kth_and_return_indices
from spotpython.hyperparameters.values import (
    get_control_key_value,
    get_ith_hyperparameter_name_from_fun_control,
)
import plotly.graph_objects as go
from typing import Callable
from spotpython.utils.numpy2json import NumpyEncoder
from spotpython.utils.file import load_result

# Setting up the backend to use QtAgg
# matplotlib.use("TkAgg")
# matplotlib.use("Agg")


logger = logging.getLogger(__name__)
# configure the handler and formatter as needed
py_handler = logging.FileHandler(f"{__name__}.log", mode="w")
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
# add formatter to the handler
py_handler.setFormatter(py_formatter)
# add handler to the logger
logger.addHandler(py_handler)


class Spot:
    """
    Spot base class to handle the following tasks in a uniform manner:

    * Getting and setting parameters. This is done via the `Spot` initialization.
    * Running surrogate based hyperparameter optimization. After the class is initialized, hyperparameter tuning
    runs can be performed via the `run` method.
    * Displaying information. The `plot` method can be used for visualizing results. The `print` methods summarizes
    information about the tuning run.

    The `Spot` class is built in a modular manner. It combines the following components:

        1. Fun (objective function)
        2. Design (experimental design)
        3. Optimizer to be used on the surrogate model
        4. Surrogate (model)

    For each of the components different implementations can be selected and combined.
    Internal components are selected as default.
    These can be replaced by components from other packages, e.g., scikit-learn or scikit-optimize.

    Args:
        fun (Callable):
            objective function
        fun_control (Dict[str, Union[int, float]]):
            objective function information stored as a dictionary.
            Default value is `fun_control_init()`.
        design (object):
            experimental design. If `None`, spotpython's `SpaceFilling` is used.
            Default value is `None`.
        design_control (Dict[str, Union[int, float]]):
            experimental design information stored as a dictionary.
            Default value is `design_control_init()`.
        optimizer (object):
            optimizer on the surrogate. If `None`, `scipy.optimize`'s `differential_evolution` is used.
            Default value is `None`.
        optimizer_control (Dict[str, Union[int, float]]):
            information about the optimizer stored as a dictionary.
            Default value is `optimizer_control_init()`.
        surrogate (object):
            surrogate model. If `None`, spotpython's `kriging` is used. Default value is `None`.
        surrogate_control (Dict[str, Union[int, float]]):
            surrogate model information stored as a dictionary.
            Default value is `surrogate_control_init()`.

    Returns:
        (NoneType): None

    Note:
        Description in the source code refers to [bart21i]:
        Bartz-Beielstein, T., and Zaefferer, M. Hyperparameter tuning approaches.
        In Hyperparameter Tuning for Machine and Deep Learning with R - A Practical Guide,
        E. Bartz, T. Bartz-Beielstein, M. Zaefferer, and O. Mersmann, Eds. Springer, 2022, ch. 4, pp. 67–114.

    Examples:
        >>> import numpy as np
            from math import inf
            from spotpython.spot import spot
            from spotpython.utils.init import (
                fun_control_init,
                design_control_init,
                surrogate_control_init,
                optimizer_control_init)
            def objective_function(X, fun_control=None):
                if not isinstance(X, np.ndarray):
                    X = np.array(X)
                if X.shape[1] != 2:
                    raise Exception
                x0 = X[:, 0]
                x1 = X[:, 1]
                y = x0**2 + 10*x1**2
                return y
            fun_control = fun_control_init(
                        lower = np.array([0, 0]),
                        upper = np.array([10, 10]),
                        fun_evals=8,
                        fun_repeats=1,
                        max_time=inf,
                        noise=False,
                        tolerance_x=0,
                        ocba_delta=0,
                        var_type=["num", "num"],
                        infill_criterion="ei",
                        n_points=1,
                        seed=123,
                        log_level=20,
                        show_models=False,
                        show_progress=True)
            design_control = design_control_init(
                        init_size=5,
                        repeats=1)
            surrogate_control = surrogate_control_init(
                        model_optimizer=differential_evolution,
                        model_fun_evals=10000,
                        min_theta=-3,
                        max_theta=3,
                        n_theta=2,
                        theta_init_zero=False,
                        n_p=1,
                        optim_p=False,
                        var_type=["num", "num"],
                        metric_factorial="canberra",
                        seed=124)
            optimizer_control = optimizer_control_init(
                        max_iter=1000,
                        seed=125)
            spot = spot.Spot(fun=objective_function,
                        fun_control=fun_control,
                        design_control=design_control,
                        surrogate_control=surrogate_control,
                        optimizer_control=optimizer_control)
            spot.run()
            spot.plot_progress()
            spot.plot_contour(i=0, j=1)
            spot.plot_importance()
    """

    def __str__(self):
        return self.__class__.__name__

    def __init__(
        self,
        design: object = None,
        design_control: dict = design_control_init(),
        fun: Callable = None,
        fun_control: dict = fun_control_init(),
        optimizer: object = None,
        optimizer_control: dict = optimizer_control_init(),
        surrogate: object = None,
        surrogate_control: dict = surrogate_control_init(),
    ):
        self.fun_control = fun_control
        self.design_control = design_control
        self.optimizer_control = optimizer_control
        self.surrogate_control = surrogate_control

        # small value:
        self.eps = sqrt(spacing(1))

        self._set_fun(fun)

        self._set_bounds_and_dim()

        # Random number generator:
        self.rng = default_rng(self.fun_control["seed"])
        set_all_seeds(self.fun_control["seed"])

        self._set_var_type()

        self._set_var_name()

        # Reduce dim based on lower == upper logic:
        # modifies lower, upper, var_type, and var_name
        self.to_red_dim()

        # Additional self attributes updates:
        self._set_additional_attributes()

        # Bounds are internal, because they are functions of self.lower and self.upper
        # and used by the optimizer:
        de_bounds = []
        for j in range(self.lower.size):
            de_bounds.append([self.lower[j], self.upper[j]])
        self.de_bounds = de_bounds

        self._design_setup(design)

        self._optimizer_setup(optimizer)

        self._surrogate_control_setup()

        # The writer (Tensorboard) must be initialized before the surrogate model,
        # because the writer is passed to the surrogate model:
        self._init_spot_writer()

        self._surrogate_setup(surrogate)

        if self.fun_control.get("save_experiment"):
            self.save_experiment(verbosity=self.verbosity)

        logger.setLevel(self.log_level)
        logger.info(f"Starting the logger at level {self.log_level} for module {__name__}:")
        logger.debug("In Spot() init(): fun_control: %s", self.fun_control)
        logger.debug("In Spot() init(): design_control: %s", self.design_control)
        logger.debug("In Spot() init(): optimizer_control: %s", self.optimizer_control)
        logger.debug("In Spot() init(): surrogate_control: %s", self.surrogate_control)
        logger.debug("In Spot() init(): self.get_spot_attributes_as_df(): %s", self.get_spot_attributes_as_df())

    def _set_fun(self, fun):
        """Set the objective function.

        Args:
            fun (Callable): objective function

        Returns:
            (NoneType): None

        Raises:
            Exception: No objective function specified.
            Exception: Objective function is not callable

        """
        self.fun = fun
        if self.fun is None:
            raise Exception("No objective function specified.")
        if not callable(self.fun):
            raise Exception("Objective function is not callable.")

    def _set_bounds_and_dim(self) -> None:
        """
        Set the lower and upper bounds and the number of dimensions.

        Returns:
            (NoneType): None

        """
        # lower attribute updates:
        # if lower is in the fun_control dictionary, use the value of the key "lower" as the lower bound
        if get_control_key_value(control_dict=self.fun_control, key="lower") is not None:
            self.lower = get_control_key_value(control_dict=self.fun_control, key="lower")
        # Number of dimensions is based on lower
        self.k = self.lower.size

        # upper attribute updates:
        # if upper is in fun_control dictionary, use the value of the key "upper" as the upper bound
        if get_control_key_value(control_dict=self.fun_control, key="upper") is not None:
            self.upper = get_control_key_value(control_dict=self.fun_control, key="upper")

    def _set_var_type(self) -> None:
        """
        Set the variable types. If the variable types are not specified,
        all variable types are forced to 'num'.
        """
        self.var_type = self.fun_control["var_type"]
        # Force numeric type as default in every dim:
        # assume all variable types are "num" if "num" is
        # specified less than k times
        if len(self.var_type) < self.k:
            self.var_type = self.var_type * self.k
            logger.warning("All variable types forced to 'num'.")

    def _set_var_name(self) -> None:
        """
        Set the variable names. If the variable names are not specified,
        all variable names are set to x0, x1, x2, ...
        """
        self.var_name = self.fun_control["var_name"]
        # use x0, x1, ... as default variable names:
        if self.var_name is None:
            self.var_name = ["x" + str(i) for i in range(len(self.lower))]

    def _set_additional_attributes(self) -> None:
        """
        Set additional attributes based on the fun_control dictionary
        """
        self.fun_evals = self.fun_control["fun_evals"]
        self.fun_repeats = self.fun_control["fun_repeats"]
        self.max_time = self.fun_control["max_time"]
        self.noise = self.fun_control["noise"]
        self.tolerance_x = self.fun_control["tolerance_x"]
        self.ocba_delta = self.fun_control["ocba_delta"]
        self.log_level = self.fun_control["log_level"]
        self.show_models = self.fun_control["show_models"]
        self.show_progress = self.fun_control["show_progress"]
        self.infill_criterion = self.fun_control["infill_criterion"]
        self.n_points = self.fun_control["n_points"]
        self.max_surrogate_points = self.fun_control["max_surrogate_points"]
        self.progress_file = self.fun_control["progress_file"]
        self.tkagg = self.fun_control["tkagg"]
        if self.tkagg:
            matplotlib.use("TkAgg")
        self.verbosity = self.fun_control["verbosity"]

        # Internal attributes:
        self.X = None
        self.y = None

        # Logging information:
        self.counter = 0
        self.min_y = None
        self.min_X = None
        self.min_mean_X = None
        self.min_mean_y = None
        self.mean_X = None
        self.mean_y = None
        self.var_y = None
        self.y_mo = None

    def _design_setup(self, design) -> None:
        """
        Design related information:
        If no design is specified, use the internal spacefilling design.
        """
        self.design = design
        if self.design is None:
            self.design = SpaceFilling(k=self.k, seed=self.fun_control["seed"])

    def _optimizer_setup(self, optimizer) -> None:
        """
        Optimizer setup. If no optimizer is specified, use Differential Evolution.
        """
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = optimize.differential_evolution

    def _surrogate_control_setup(self) -> None:
        self.surrogate_control.update({"var_type": self.var_type})
        # Surrogate control updates:
        # The default value for `method` from the surrogate_control dictionary
        # based on surrogate_control.init() is None. This value is updated
        # to the value of the key "method" from the fun_control dictionary.
        # If the value is set (i.e., not None), it is not updated.
        if self.surrogate_control["method"] is None:
            self.surrogate_control.update({"method": self.fun_control.method})
        if self.surrogate_control["model_fun_evals"] is None:
            self.surrogate_control.update({"model_fun_evals": self.optimizer_control["max_iter"]})
        # self.optimizer is not None here. If 1) the key "model_optimizer"
        # is still None or 2) a user specified optimizer is provided, update the value of
        # the key "model_optimizer" to the value of self.optimizer.
        if self.surrogate_control["model_optimizer"] is None or self.optimizer is not None:
            self.surrogate_control.update({"model_optimizer": self.optimizer})

        # if self.surrogate_control["n_theta"] is a string and == isotropic, use 1 theta value:
        if isinstance(self.surrogate_control["n_theta"], str):
            if self.surrogate_control["n_theta"] == "anisotropic":
                self.surrogate_control.update({"n_theta": self.k})
            else:
                # case "isotropic":
                self.surrogate_control.update({"n_theta": 1})
        if isinstance(self.surrogate_control["n_theta"], int):
            if self.surrogate_control["n_theta"] > 1:
                self.surrogate_control.update({"n_theta": self.k})

    def _surrogate_setup(self, surrogate) -> None:
        # Surrogate related information:
        self.surrogate = surrogate
        # If no surrogate model is specified, use the internal
        # spotpython kriging surrogate:
        if self.surrogate is None:
            # Call kriging with surrogate_control parameters:
            self.surrogate = Kriging(
                method=self.surrogate_control["method"],
                var_type=self.surrogate_control["var_type"],
                seed=self.surrogate_control["seed"],
                model_optimizer=self.surrogate_control["model_optimizer"],
                model_fun_evals=self.surrogate_control["model_fun_evals"],
                min_theta=self.surrogate_control["min_theta"],
                max_theta=self.surrogate_control["max_theta"],
                n_theta=self.surrogate_control["n_theta"],
                theta_init_zero=self.surrogate_control["theta_init_zero"],
                p_val=self.surrogate_control["p_val"],
                n_p=self.surrogate_control["n_p"],
                optim_p=self.surrogate_control["optim_p"],
                min_Lambda=self.surrogate_control["min_Lambda"],
                max_Lambda=self.surrogate_control["max_Lambda"],
                log_level=self.log_level,
                spot_writer=self.spot_writer,
                counter=self.design_control["init_size"] * self.design_control["repeats"] - 1,
                metric_factorial=self.surrogate_control["metric_factorial"],
            )

    def get_spot_attributes_as_df(self) -> pd.DataFrame:
        """Get all attributes of the spot object as a pandas dataframe.

        Returns:
            (pandas.DataFrame): dataframe with all attributes of the spot object.

        Examples:
            >>> import numpy as np
                from math import inf
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import spot
                from spotpython.utils.init import (
                    fun_control_init, design_control_init
                    )
                # number of initial points:
                ni = 7
                # number of points
                n = 10
                fun = Analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1]),
                    upper = np.array([1]),
                    fun_evals=n)
                design_control=design_control_init(init_size=ni)
                spot_1 = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control,)
                spot_1.run()
                df = spot_1.get_spot_attributes_as_df()
            df
                df
                    Attribute Name                                    Attribute Value
                0                   X  [[-0.3378148180708981], [0.698908280342222], [...
                1           all_lower                                               [-1]
                2           all_upper                                                [1]
                3        all_var_name                                               [x0]
                4        all_var_type                                              [num]
                5             counter                                                 10
                6           de_bounds                                          [[-1, 1]]
                7              design  <spotpython.design.spacefilling.SpaceFilling o...
                8      design_control                     {'init_size': 7, 'repeats': 1}
                9                 eps                                                0.0
                10        fun_control                         {'sigma': 0, 'seed': None}
                11          fun_evals                                                 10
                12        fun_repeats                                                  1
                13              ident                                            [False]
                14   infill_criterion                                                  y
                15                  k                                                  1
                16          log_level                                                 50
                17              lower                                               [-1]
                18           max_time                                                inf
                19             mean_X                                               None
                20             mean_y                                               None
                21              min_X                           [1.5392206722432657e-05]
                22         min_mean_X                                               None
                23         min_mean_y                                               None
                24              min_y                                                0.0
                25           n_points                                                  1
                26              noise                                               True
                27         ocba_delta                                                  0
                28  optimizer_control                    {'max_iter': 1000, 'seed': 125}
                29            red_dim                                              False
                30                rng                                   Generator(PCG64)
                31               seed                                                123
                32        show_models                                              False
                33      show_progress                                               True
                34        spot_writer                                               None
                35          surrogate  <spotpython.build.kriging.Kriging object at 0x...
                36  surrogate_control  {'method': "regession", 'model_optimizer': <function ...
                37        tolerance_x                                                  0
                38              upper                                                [1]
                39           var_name                                               [x0]
                40           var_type                                              [num]
                41              var_y                                               None
                42                  y  [0.11411885130827397, 0.48847278433092195, 0.0...

        """

        attributes = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        values = [getattr(self, attr) for attr in attributes]
        df = pd.DataFrame({"Attribute Name": attributes, "Attribute Value": values})
        return df

    def to_red_dim(self) -> None:
        """
        Reduces the dimensionality of the optimization problem by removing dimensions
        where lower and upper bounds are equal, indicating that those variables are fixed.
        This function modifies the lower bounds, upper bounds, variable types, and variable names
        by filtering out the non-varying dimensions. If any dimension is reduced, the `red_dim` attribute
        is set to True, and the count of dimensions (`k`) is updated accordingly.

        Args:
            self (object): Spot object

        Returns:
            (NoneType): None

        Attributes:
            self.lower (numpy.ndarray):
                lower bound
            self.upper (numpy.ndarray):
                upper bound
            self.var_type (List[str]):
                list of variable types
            self.ident (numpy.ndarray):
                array of boolean values indicating fixed dimensions
            self.red_dim (bool):
                True if dimensions are reduced, False otherwise. Checks if any dimension is fixed.
            self.all_lower (numpy.ndarray):
                backup of the original lower bounds
            self.all_upper (numpy.ndarray):
                backup of the original upper bounds

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import spot
                from spotpython.utils.init import fun_control_init, surrogate_control_init, design_control_init
                lower = np.array([-1, -1, 0, 0])
                upper = np.array([1, -1, 0, 5])  # Second and third dimensions are fixed
                fun_evals = 10
                var_type = ['float', 'int', 'float', 'int']
                var_name = ['x1', 'x2', 'x3', 'x4']
                spot_instance = spot.Spot(
                    # Assuming Spot takes fun, fun_control, design_control as arguments
                    fun = Analytical().fun_sphere,  # Replace with appropriate function
                    fun_control=fun_control_init(lower=lower, upper=upper, fun_evals=fun_evals, show_progress=True, log_level=50),
                )
                spot_instance.var_type = var_type
                spot_instance.var_name = var_name
                spot_instance.to_red_dim()
                # Assert: Check if dimensions were reduced correctly
                assert spot_instance.lower.size == 2, "Expected size of lower to be 2 after reduction"
                assert spot_instance.upper.size == 2, "Expected size of upper to be 2 after reduction"
                assert len(spot_instance.var_type) == 2, "Expected size of var_type to be 2 after reduction"
                assert spot_instance.k == 2, "Expected k to reflect the reduced dimensions"
                # Check remaining values
                expected_lower = np.array([-1, 0])
                expected_upper = np.array([1, 5])
                expected_var_type = ['float', 'int']
                # there are two remaining variables, they are named x1 and x2
                expected_var_name = ['x1', 'x2']
                np.testing.assert_array_equal(spot_instance.lower, expected_lower)
                np.testing.assert_array_equal(spot_instance.upper, expected_upper)
                assert spot_instance.var_type == expected_var_type
                assert spot_instance.var_name == expected_var_name
        """
        # Backup of the original values:
        self.all_lower = self.lower
        self.all_upper = self.upper
        # Select only lower != upper:
        self.ident = (self.upper - self.lower) == 0
        # Determine if dimension is reduced:
        self.red_dim = self.ident.any()
        # Modifications:
        # Modify lower and upper:
        self.lower = self.lower[~self.ident]
        self.upper = self.upper[~self.ident]
        # Modify k (number of dim):
        self.k = self.lower.size
        # Filter out types and names corresponding to non-varying dimensions
        if self.var_type is not None:
            self.all_var_type = self.var_type.copy()
            self.var_type = [vtype for vtype, fixed in zip(self.all_var_type, self.ident) if not fixed]

        if self.var_name is not None:
            self.all_var_name = self.var_name.copy()
            self.var_name = [vname for vname, fixed in zip(self.all_var_name, self.ident) if not fixed]

    def to_all_dim(self, X0: np.ndarray) -> np.ndarray:
        """
        Expands reduced-dimensional design points back to their full-dimensional representation
        by reinserting fixed values for dimensions that were removed during the dimensionality
        reduction process.
        When `to_red_dim()` is called, dimensions where the lower and upper bounds are equal are
        removed from the design points. `to_all_dim()` reverses this process by adding back these fixed
        dimensions with their respective fixed values.

        Args:
            X0 (numpy.ndarray): reduced dimension design points

        Returns:
            (numpy.ndarray): full dimension design points

        Atributes:
            self.ident (numpy.ndarray):
                array of boolean values indicating fixed dimensions
            self.all_lower (numpy.ndarray):
                backup of the original lower bounds.

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import spot
                from spotpython.utils.init import fun_control_init, surrogate_control_init, design_control_init
                lower = np.array([-1, -1, 0, 0])
                upper = np.array([1, -1, 0, 5])  # Second and third dimensions are fixed
                fun_evals = 10
                var_type = ['float', 'int', 'float', 'int']
                var_name = ['x1', 'x2', 'x3', 'x4']
                spot_instance = spot.Spot(
                    fun = Analytical().fun_sphere,
                    fun_control=fun_control_init(lower=lower, upper=upper, fun_evals=fun_evals)
                )
                X0 = np.array([[2.5, 3.5], [4.5, 5.5]])
                X_full_dim = spot_instance.to_all_dim(X0)
                print(X_full_dim)
                    [[ 2.5 -1.   0.   3.5]
                    [ 4.5 -1.   0.   5.5]]
        """
        # Number of design points (samples):
        n = X0.shape[0]
        # Number of dimensions:
        k = len(self.ident)
        # Initialize full dimension design points:
        X = np.zeros((n, k))
        # The following code was modified in 0.20.5:
        # Index for navigating X0's compressed dimension
        reduced_index = 0
        # Iterate through each dimension, reconstructing full dimensionality
        for i in range(k):
            if self.ident[i]:
                # Assign fixed dimension values using stored lower bounds
                X[:, i] = self.all_lower[i]
            else:
                # Assign variable dimension values from the reduced array
                X[:, i] = X0[:, reduced_index]
                # Move to the next variable dimension in the compact X0 array
                reduced_index += 1
        return X

    def to_all_dim_if_needed(self, X: np.ndarray) -> np.array:
        """
        Conditionally expand reduced-dimensional design points back to their full-dimensional representation,
        if dimensionality reduction was performed.
        This method checks whether dimensionality reduction occurred (i.e., whether some dimensions were
        fixed and thus removed). If so, it uses `to_all_dim()` to restore the full-dimensional representation
        by reinserting the fixed dimensions. Otherwise, it returns the input design points unaltered.

        Args:
            X (np.ndarray): A 2D numpy array of shape (n, m), where `n` is the number of samples, and `m`
                            corresponds to the reduced or full number of dimensions depending on the
                            `red_dim` status.

        Returns:
            np.ndarray: A 2D numpy array of shape (n, k). If `red_dim` is True, `k` will be the full number
                        of dimensions (including both fixed and variable). If `red_dim` is False, `k` is
                        identical to `m`.

        Attributes:
            self.red_dim (bool): A boolean attribute indicating if dimensionality was reduced
                                (True if dimensions were reduced, False otherwise).
        """

        if self.red_dim:
            return self.to_all_dim(X)
        else:
            return X

    def get_new_X0(self) -> np.array:
        """
        Generate new design points for the optimization process.
        This method attempts to suggest and repair new design points using the surrogate model
        and experimental design techniques. If no valid new points are found within the specified
        tolerance, a new experimental design is generated.

        Calls `suggest_new_X()` and repairs the new design points, e.g.,
        by `repair_non_numeric()` and `selectNew()`.

        Returns:
            np.ndarray: New design points, possibly repeated according to `self.fun_repeats`.

        Attributes:
            self.design (object): An experimental design object used to generate new points
            self.n_points (int): The expected number of new points
            self.fun_repeats (int): The number of times to repeat new points
            self.tolerance_x (float): Minimum distance required between new and existing solutions
            self.var_type (List[str]): Variable types for the design points
            self.X (np.ndarray): Existing solution points
            self.k (int): Number of dimensions
            self.fun_control (Dict): Control parameters for the function
            self.counter (int): Iteration counter

        Notes:
            - If no new valid design points are suggested, the function resorts
              to a space-filling design technique to generate the required points.

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.utils.init import (
                    fun_control_init,  design_control_init
                    )
                from spotpython.spot import spot
                from spotpython.utils.init import fun_control_init
                # number of initial points:
                ni = 3
                X_start = np.array([[0, 1], [1, 0], [1, 1], [1, 1]])
                fun = Analytical().fun_sphere
                fun_control = fun_control_init(
                            n_points=10,
                            ocba_delta=0,
                            lower = np.array([-1, -1]),
                            upper = np.array([1, 1])
                )
                design_control=design_control_init(init_size=ni)
                S = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control,
                )
                S.initialize_design(X_start=X_start)
                S.update_stats()
                S.fit_surrogate()
                X0 = S.get_new_X0()
                assert X0.shape[0] == S.n_points
                assert X0.shape[1] == S.lower.size
                # assert new points are in the interval [lower, upper]
                assert np.all(X0 >= S.lower)
                assert np.all(X0 <= S.upper)
                # print using 20 digits precision
                np.set_printoptions(precision=20)
                print(f"X0: {X0}")
                X0: [[-0.43905273463270317 -0.20947824142606025]
                    [-0.4390526520612617  -0.20947735118625146]
                    [-0.4390526516559971  -0.20947735345727678]
                    [-0.4390526491133424  -0.20947735153559494]
                    [-0.43905264887606393 -0.209477347335596  ]
                    [-0.43905264815296263 -0.20947734884431773]
                    [-0.4390526481478378  -0.2094773501907511 ]
                    [-0.43905264791185933 -0.20947734931732975]
                    [-0.43905264783691894 -0.20947734910961185]
                    [-0.4390526473921517  -0.2094773511154602 ]]
        """
        # Try to generate self.fun_repeats new X0 points:
        X0 = self.suggest_new_X()
        # Repair non-numeric variables based on their types
        X0 = repair_non_numeric(X0, self.var_type)
        # Condition: select only X0 that have min distance self.tolerance_x
        # to existing solutions
        X0, X0_ind = selectNew(A=X0, X=self.X, tolerance=self.tolerance_x)
        if X0.shape[0] > 0:
            # If valid new points are found, repeat them as specified
            # There are X0 that fullfil the condition.
            # Note: The number of new X0 can be smaller than self.n_points!
            logger.debug("XO values are new: %s %s", X0_ind, X0)
            return repeat(X0, self.fun_repeats, axis=0)
        # If no X0 found, then generate self.n_points new solutions:
        else:
            self.design = SpaceFilling(k=self.k, seed=self.fun_control["seed"] + self.counter)
            X0 = self.generate_design(size=self.n_points, repeats=self.design_control["repeats"], lower=self.lower, upper=self.upper)
            X0 = repair_non_numeric(X0, self.var_type)
            logger.warning("No new XO found on surrogate. Generate new solution %s", X0)
            return X0

    def run(self, X_start: np.ndarray = None) -> Spot:
        """
        Run the surrogate based optimization.
        The optimization process is controlled by the following steps:
            1. Initialize design
            2. Update stats
            3. Fit surrogate
            4. Update design
            5. Update stats
            6. Update writer
            7. Fit surrogate
            8. Show progress if needed

        Args:
            X_start (numpy.ndarray, optional):
                initial design. The initial design must have shape (n, k), where n is the number of points and k is the number of dimensions. Defaults to None.

        Returns:
            Spot: The `Spot` instance configured and updated based on the optimization process.

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import Spot
                from spotpython.utils.init import (
                    fun_control_init, design_control_init
                    )
                # number of initial points:
                ni = 7
                # start point X_0
                X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                fun = Analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1, -1]),
                    upper = np.array([1, 1]))
                design_control=design_control_init(init_size=ni)
                S = Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control,)
                S.run(X_start=X_start)
                    spotpython tuning: 0.0 [########--] 80.00%
                    spotpython tuning: 0.0 [#########-] 86.67%
                    spotpython tuning: 0.0 [#########-] 93.33%
                    spotpython tuning: 0.0 [##########] 100.00% Done...
            >>> S.print_results()
                min y: 0.0
                x0: 0.0
                x1: 0.0
            >>> S.X
                array([[ 0.0000000000000000e+00,  0.0000000000000000e+00],
                [ 0.0000000000000000e+00,  1.0000000000000000e+00],
                [ 1.0000000000000000e+00,  0.0000000000000000e+00],
                [ 1.0000000000000000e+00,  1.0000000000000000e+00],
                [-9.0924338949946959e-01, -1.5823457680063502e-01],
                [-2.0581710650646035e-01, -4.8124908877104844e-01],
                [ 9.4974117111856260e-01, -9.4631271618736390e-01],
                [-1.2095571372201608e-01,  6.3835886343683867e-02],
                [-6.6278701759800063e-01,  1.7431637339680406e-01],
                [ 2.8200844136299108e-01,  9.3001011398034406e-01],
                [ 4.7878811540073962e-01,  6.5321058189282999e-01],
                [ 1.5404061268479530e-04,  4.1895410759355553e-03],
                [-1.7027205448129213e-04,  4.7698567182254507e-03],
                [-4.4080972128058849e-04,  5.2785168039883147e-03],
                [ 3.7700880321788425e-03,  1.8909833144458731e-02]])
            >>> S.y
                array([0.0000000000000000e+00, 1.0000000000000000e+00,
                1.0000000000000000e+00, 2.0000000000000000e+00,
                8.5176172264376016e-01, 2.7396136677365612e-01,
                1.7975160489355650e+00, 1.8705305067286033e-02,
                4.6967282873066640e-01, 9.4444757310571603e-01,
                6.5592212374576153e-01, 1.7575982937307560e-05,
                2.2780525684937748e-05, 2.8057052860362483e-05,
                3.7179535332164810e-04])

        """
        #
        PREFIX = self.fun_control["PREFIX"]
        filename = get_result_filename(PREFIX)
        if os.path.exists(filename) and not self.fun_control.get("force_run"):
            # print a warning and load the result
            print(f"Result file {filename} exists. Loading the result.")
            S = load_result(filename=filename)
            self._copy_from(S)
            return self
        else:
            self.initialize_design(X_start)
            self.update_stats()
            self.fit_surrogate()
            if self.verbosity > 0:
                print("\n------ Starting optimization on the Surrogate for the given Budget ------\n")
            timeout_start = time.time()
            while self.should_continue(timeout_start):
                self.update_design()
                self.update_stats()
                self.update_writer()
                self.fit_surrogate()
                self.show_progress_if_needed(timeout_start)

            if hasattr(self, "spot_writer") and self.spot_writer is not None:
                self.spot_writer.flush()
                self.spot_writer.close()
            if self.fun_control.get("db_dict_name") is not None:
                self._write_db_dict()

            if self.fun_control.get("save_result"):
                self.save_result(verbosity=self.verbosity)
            return self

    def _copy_from(self, other) -> None:
        """Copy attributes from another object.
        This method copies all attributes from the `other` object to the current
        object (`self`). It assumes that both objects are instances of a class
        that share similar attributes.

        Args:
            other: An instance of a class from which attributes will be copied to
            the current instance.

        """
        for attr in other.__dict__:
            setattr(self, attr, getattr(other, attr))

    def initialize_design(self, X_start=None) -> None:
        """
        Initialize design. Generate and evaluate initial design.
        If `X_start` is not `None`, append it to the initial design.
        Therefore, the design size is `init_size` + `X_start.shape[0]`.

        Args:
            X_start (numpy.ndarray, optional):
                initial design. Must be of shape (n, k), where n is the number
                of points and k is the number of dimensions. Defaults to None.

        Attributes:
            self.X (numpy.ndarray): initial design
            self.y (numpy.ndarray): initial design values

        Note:
            * If `X_start` is has the wrong shape, it is ignored.

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import spot
                from spotpython.utils.init import (
                    fun_control_init,  design_control_init
                    )
                # number of initial points:
                ni = 7
                # start point X_0
                X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                fun = Analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1, -1]),
                    upper = np.array([1, 1]))
                design_control=design_control_init(init_size=ni)
                S = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control,)
                S.initialize_design(X_start=X_start)
                print(f"S.X: {S.X}")
                    S.X: [[ 0.          0.        ]
                        [ 0.          1.        ]
                        [ 1.          0.        ]
                        [ 1.          1.        ]
                        [-0.90924339 -0.15823458]
                        [-0.20581711 -0.48124909]
                        [ 0.94974117 -0.94631272]
                        [-0.12095571  0.06383589]
                        [-0.66278702  0.17431637]
                        [ 0.28200844  0.93001011]
                        [ 0.47878812  0.65321058]]
                print(f"S.y: {S.y}")
                        S.y: [0.         1.         1.         2.         0.85176172 0.27396137
                            1.79751605 0.01870531 0.46967283 0.94444757 0.65592212]
        """
        self.initialize_design_matrix(X_start)

        self.evaluate_initial_design()

        self.write_initial_tensorboard_log()

    def initialize_design_matrix(self, X_start=None) -> None:
        """
        Initialize the design matrix for the optimization process.
        This method generates an initial design matrix, optionally
        appending any provided starting points (`X_start`). The resulting
        design matrix is sanitized for non-numeric values and stored in `self.X`.

        Args:
            X_start (numpy.ndarray, optional): User-provided starting points
                for the design. Shape should be (n=n_samples, k=n_features).
                Defaults to None.

        Returns:
            numpy.ndarray: The design matrix that combines the generated design
                        with the provided starting points.

        Raises:
            Exception: If the resulting design matrix has zero rows.

        Notes:
            * If `X_start` is not in the expected shape, it is ignored.

        Examples:
            >>> import numpy as np
                from spotpython.fun import Analytical
                from spotpython.spot import Spot
                from spotpython.utils.init import fun_control_init
                fun_control = fun_control_init(
                    tensorboard_log=True,
                    TENSORBOARD_CLEAN=True,
                    lower = np.array([-1]),
                    upper = np.array([1])
                    )
                fun = Analytical().fun_sphere
                S = Spot(fun=fun,
                            fun_control=fun_control,
                            )
                X_start = np.array([[0.5, 0.5], [0.4, 0.4]])
                design_matrix = S.initialize_design_matrix(X_start)
                print(f"Design matrix: {design_matrix}")
                    Design matrix: [[ 0.1         0.2       ]
                    [ 0.3         0.4       ]
                    [ 0.86352963  0.7892358 ]
                    [-0.24407197 -0.83687436]
                    [ 0.36481882  0.8375811 ]
                    [ 0.415331    0.54468512]
                    [-0.56395091 -0.77797854]
                    [-0.90259409 -0.04899292]
                    [-0.16484832  0.35724741]
                    [ 0.05170659  0.07401196]
                    [-0.78548145 -0.44638164]
                    [ 0.64017497 -0.30363301]]
        """
        if self.design_control["init_size"] > 0:
            X0 = self.generate_design(
                size=self.design_control["init_size"],
                repeats=self.design_control["repeats"],
                lower=self.lower,
                upper=self.upper,
            )

        if X_start is not None:
            if not isinstance(X_start, np.ndarray):
                X_start = np.array(X_start)
            X_start = np.atleast_2d(X_start)
            try:
                if self.design_control["init_size"] > 0:
                    X0 = np.append(X_start, X0, axis=0)
                else:
                    X0 = X_start
            except ValueError:
                logger.warning("X_start has wrong shape. Ignoring it.")

        if X0.shape[0] == 0:
            raise Exception("X0 has zero rows. Check design_control['init_size'] or X_start.")

        self.X = repair_non_numeric(X0, self.var_type)

    def _store_mo(self, y_mo) -> None:
        # store y_mo in self.y_mo (append new values) if mo, otherwise self.y_mo is None
        if self.y_mo is None and y_mo.ndim == 2:
            self.y_mo = y_mo
        else:  # append new values
            # before stacking the arrays, check if the number of columns is the same in the mo case
            if y_mo.ndim == 2 and self.y_mo.ndim == 2:
                if self.y_mo.shape[1] != y_mo.shape[1]:
                    print(f"Shape of y_mo: {y_mo.shape}")
                    print(f"y_mo: {y_mo}")
                    print(f"Shape of self.y_mo: {self.y_mo.shape}")
                    print(f"self.y_mo: {self.y_mo}")
                    raise ValueError(f"Number of columns (objectives) in y_mo ({y_mo.shape[1]}) " f"does not match the number of columns in self.y_mo ({self.y_mo.shape[1]})")
                self.y_mo = np.vstack((self.y_mo, y_mo))

    def _mo2so(self, y_mo) -> None:
        """
        Converts multi-objective values to a single-objective value by applying a user-defined
        function from ``fun_control['fun_mo2so']``. If no user-defined function is given, the
        values in the first objective row are used.

        This method is called after the objective function evaluation (i.e., after ``self.fun()``).
        It typically returns a 1D array with the single-objective values.

        Args:
            y_mo (numpy.ndarray):
                If multi-objective values are present, this is an array of shape (n, m), where ``m`` is
                the number of objectives and ``n`` is the number of data points.
                Otherwise, it is an array of shape (n,) with single-objective values.
        Returns:
            numpy.ndarray:
                A 1D array of shape (n,) with single-objective values.

        """
        n, m = get_shape(y_mo)
        self._store_mo(y_mo)
        # do not use m as a condition, because m can be None, use ndim instead
        if y_mo.ndim == 2:
            if self.fun_control["fun_mo2so"] is not None:
                y0 = self.fun_control["fun_mo2so"](y_mo)
            else:
                # Select the first column of an (n,m) array
                if y_mo.size > 0:
                    y0 = y_mo[:, 0]
                else:
                    y0 = y_mo
        else:
            y0 = y_mo
        return y0

    def evaluate_initial_design(self) -> None:
        """
        Evaluate the initial design.

        This method evaluates the initial design matrix `X0` by applying the objective function
        and handling NaN values. The results are stored in `self.X` and `self.y`.

        Raises:
            ValueError: If the resulting design matrix has zero rows after removing NaN values.

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import Spot
                from spotpython.utils.init import fun_control_init
                fun_control = fun_control_init(
                    lower=np.array([-1, -1]),
                    upper=np.array([1, 1])
                )
                fun = Analytical().fun_sphere
                S = Spot(fun=fun, fun_control=fun_control)
                X0 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                S.initialize_design_matrix(X_start=X0)
                S.evaluate_initial_design()
                print(f"S.X: {S.X}")
                print(f"S.y: {S.y}")
                    S.X: [[ 0.          0.        ]
                    [ 0.          1.        ]
                    [ 1.          0.        ]
                    [ 1.          1.        ]
                    [ 0.86352963  0.7892358 ]
                    [-0.24407197 -0.83687436]
                    [ 0.36481882  0.8375811 ]
                    [ 0.415331    0.54468512]
                    [-0.56395091 -0.77797854]
                    [-0.90259409 -0.04899292]
                    [-0.16484832  0.35724741]
                    [ 0.05170659  0.07401196]
                    [-0.78548145 -0.44638164]
                    [ 0.64017497 -0.30363301]]
                    S.y: [0.         1.         1.         2.         1.36857656 0.75992983
                    0.83463487 0.46918172 0.92329124 0.8170764  0.15480068 0.00815134
                    0.81623768 0.502017  ]
        """
        # check that self.X has at leat one row and is not None
        if self.X is None or self.X.shape[0] == 0:
            raise ValueError("The design matrix has zero rows. Check design_control['init_size'] or X_start.")

        X_all = self.to_all_dim_if_needed(self.X)
        logger.debug("In Spot() evaluate_initial_design(), before calling self.fun: X_all: %s", X_all)
        logger.debug("In Spot() evaluate_initial_design(), before calling self.fun: fun_control: %s", self.fun_control)

        y_mo = self.fun(X=X_all, fun_control=self.fun_control)
        if self.verbosity > 1:
            print(f"y_mo as returned from fun(): {y_mo}")
            print(f"y_mo shape: {y_mo.shape}")

        #  Convert multi-objective values to single-objective values
        # TODO: Store y_mo in self.y_mo (append new values)
        self.y = self._mo2so(y_mo)
        self.y = apply_penalty_NA(self.y, self.fun_control["penalty_NA"], verbosity=self.verbosity)
        logger.debug("In Spot() evaluate_initial_design(), after calling self.fun: self.y: %s", self.y)

        # TODO: Error if only nan values are returned
        logger.debug("New y value: %s", self.y)

        self.counter = self.y.size
        self.X, self.y = remove_nan(self.X, self.y, stop_on_zero_return=True)

        if self.X.shape[0] == 0:
            raise ValueError("The resulting design matrix has zero rows after removing NaN values.")

        logger.debug("In Spot() evaluate_initial_design(), final X val, after remove nan: self.X: %s", self.X)
        logger.debug("In Spot() evaluate_initial_design(), final y val, after remove nan: self.y: %s", self.y)

    def write_initial_tensorboard_log(self) -> None:
        """Writes initial design data using the spot_writer. The spot_writer
        is a tensorboard writer that writes the data to a tensorboard file.

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import spot
                from spotpython.utils.init import fun_control_init
                fun = Analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1]),
                    upper = np.array([1])
                    )
                S = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            )
                S.initialize_design()
                S.write_initial_tensorboard_log()
                    Moving TENSORBOARD_PATH: runs/ to TENSORBOARD_PATH_OLD: runs_OLD/runs_2025_01_12_09_24_15
                    Created spot_tensorboard_path: runs/spot_logs/00_p040025_2025-01-12_09-24-15 for SummaryWriter()
        """
        if hasattr(self, "spot_writer") and self.spot_writer is not None:
            # range goes to init_size -1 because the last value is added by update_stats(),
            # which always adds the last value.
            # Changed in 0.5.9:
            for j in range(len(self.y)):
                X_j = self.X[j].copy()
                y_j = self.y[j].copy()
                config = {self.var_name[i]: X_j[i] for i in range(self.k)}
                # var_dict = assign_values(X, get_var_name(fun_control))
                # config = list(generate_one_config_from_var_dict(var_dict, fun_control))[0]
                # see: https://github.com/pytorch/pytorch/issues/32651
                # self.spot_writer.add_hparams(config, {"spot_y": y_j}, run_name=self.spot_tensorboard_path)
                self.spot_writer.add_hparams(config, {"hp_metric": y_j})
                self.spot_writer.flush()

    def update_stats(self) -> None:
        """
        Update the following stats:
        1. `min_y`, 2. `min_X`, and 3. `counter`
        If `noise` is `True`, additionally the following stats are computed:
        1. `mean_X`,  2. `mean_y`,  3. `var_y`, 4. `min_mean_X`(X value of the best mean y value so far),
        5. `min_mean_y` (best mean y value so far), and 6. `min_var_y` (ariance of the best mean y value so far).

        Args:
            self (object): Spot object

        Returns:
            (NoneType): None

        Attributes:
            self.min_y (float): minimum y value
            self.min_X (numpy.ndarray): X value of the minimum y value
            self.counter (int): number of function evaluations
            self.mean_X (numpy.ndarray): mean X values
            self.mean_y (numpy.ndarray): mean y values
            self.var_y (numpy.ndarray): variance of y values
            self.min_mean_y (float): minimum mean y value
            self.min_mean_X (numpy.ndarray): X value of the minimum mean y value

        """
        self.min_y = min(self.y)
        self.min_X = self.X[argmin(self.y)]
        self.counter = self.y.size
        self.fun_control.update({"counter": self.counter})
        # Update aggregated x and y values (if noise):
        if self.noise:
            Z = aggregate_mean_var(X=self.X, y=self.y)
            self.mean_X = Z[0]
            self.mean_y = Z[1]
            self.var_y = Z[2]
            # X value of the best mean y value so far:
            self.min_mean_X = self.mean_X[argmin(self.mean_y)]
            # best mean y value so far:
            self.min_mean_y = self.mean_y[argmin(self.mean_y)]
            # variance of the best mean y value so far:
            self.min_var_y = self.var_y[argmin(self.mean_y)]

    def fit_surrogate(self) -> None:
        """
        Fit surrogate model. The surrogate model
        is fitted to the data stored in `self.X` and `self.y`.
        It uses the generic `fit()` method of the
        surrogate model `surrogate`. The default surrogate model is
        an instance from spotpython's `Kriging` class.
        If `show_models` is `True`, the model is plotted.
        If the number of points is greater than `max_surrogate_points`,
        the surrogate model is fitted to a subset of the data points.
        The subset is selected using the `select_distant_points()` function.

        Args:
            self (object): Spot object

        Returns:
            (NoneType): None

        Attributes:
            self.surrogate (object):
                surrogate model
            self.X (numpy.ndarray):
                design points
            self.y (numpy.ndarray):
                function values
            self.max_surrogate_points (int):
                maximum number of points to fit the surrogate model
            self.show_models (bool):
                if True, the model is plotted

        Note:
            * As shown in https://sequential-parameter-optimization.github.io/Hyperparameter-Tuning-Cookbook/
            other surrogate models can be used as well.

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import spot
                from spotpython.utils.init import (
                fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
                )
                # number of initial points:
                ni = 0
                X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
                fun = analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1, -1]),
                    upper = np.array([1, 1])
                    )
                design_control=design_control_init(init_size=ni)
                S = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control,)
                S.initialize_design(X_start=X_start)
                S.update_stats()
                S.fit_surrogate()
                S.surrogate.predict(np.array([[0, 0]]))
                    array([1.49011612e-08])

        """
        logger.debug("In fit_surrogate(): self.X: %s", self.X)
        logger.debug("In fit_surrogate(): self.y: %s", self.y)
        logger.debug("In fit_surrogate(): self.X.shape: %s", self.X.shape)
        logger.debug("In fit_surrogate(): self.y.shape: %s", self.y.shape)
        X_points = self.X.shape[0]
        y_points = self.y.shape[0]
        if X_points == y_points:
            if X_points > self.max_surrogate_points:
                logger.info("Selecting distant points for surrogate fitting.")
                X_S, y_S = select_distant_points(X=self.X, y=self.y, k=self.max_surrogate_points)
            else:
                X_S = self.X
                y_S = self.y
            self.surrogate.fit(X_S, y_S)
        else:
            logger.warning("X and y have different sizes. Surrogate not fitted.")
        if self.show_models:
            self.plot_model()

    def update_design(self) -> None:
        """
        Update design. Generate and evaluate new design points.
        It is basically a call to the method `get_new_X0()`.
        If `noise` is `True`, additionally the following steps
        (from `get_X_ocba()`) are performed:
        1. Compute OCBA points.
        2. Evaluate OCBA points.
        3. Append OCBA points to the new design points.

        Args:
            self (object): Spot object

        Returns:
            (NoneType): None

        Attributes:
            self.X (numpy.ndarray): updated design
            self.y (numpy.ndarray): updated design values

        Examples:
            >>> # 1. Without OCBA points:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.utils.init import (
                    fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
                    )
                from spotpython.spot import Spot
                # number of initial points:
                ni = 0
                X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
                fun = Analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1, -1]),
                    upper = np.array([1, 1])
                    )
                design_control=design_control_init(init_size=ni)
                S = Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control,)
                S.initialize_design(X_start=X_start)
                print(f"S.X: {S.X}")
                print(f"S.y: {S.y}")
                X_shape_before = S.X.shape
                print(f"X_shape_before: {X_shape_before}")
                print(f"y_size_before: {S.y.size}")
                y_size_before = S.y.size
                S.update_stats()
                S.fit_surrogate()
                S.update_design()
                print(f"S.X: {S.X}")
                print(f"S.y: {S.y}")
                print(f"S.n_points: {S.n_points}")
                print(f"X_shape_after: {S.X.shape}")
                print(f"y_size_after: {S.y.size}")
            >>> #
            >>> # 2. Using the OCBA points:
                import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import Spot
                from spotpython.utils.init import fun_control_init, design_control_init
                # number of initial points:
                ni = 3
                X_start = np.array([[0, 1], [1, 0], [1, 1], [1, 1]])
                fun = Analytical().fun_sphere
                fun_control = fun_control_init(
                        sigma=0.02,
                        lower = np.array([-1, -1]),
                        upper = np.array([1, 1]),
                        noise=True,
                        ocba_delta=1,
                    )
                design_control=design_control_init(init_size=ni, repeats=2)
                S = Spot(fun=fun,
                            design_control=design_control,
                            fun_control=fun_control
                )
                S.initialize_design(X_start=X_start)
                print(f"S.X: {S.X}")
                print(f"S.y: {S.y}")
                X_shape_before = S.X.shape
                print(f"X_shape_before: {X_shape_before}")
                print(f"y_size_before: {S.y.size}")
                y_size_before = S.y.size
                S.update_stats()
                S.fit_surrogate()
                S.update_design()
                print(f"S.X: {S.X}")
                print(f"S.y: {S.y}")
                print(f"S.n_points: {S.n_points}")
                print(f"S.ocba_delta: {S.ocba_delta}")
                print(f"X_shape_after: {S.X.shape}")
                print(f"y_size_after: {S.y.size}")
                # compare the shapes of the X and y values before and after the update_design method
                assert X_shape_before[0] + S.ocba_delta == S.X.shape[0]
                assert X_shape_before[1] == S.X.shape[1]
                assert y_size_before + S.ocba_delta == S.y.size
                Experiment saved to 000_exp.pkl
                    S.X: [[ 0.          1.        ]
                    [ 1.          0.        ]
                    [ 1.          1.        ]
                    [ 1.          1.        ]
                    [ 0.54509876 -0.36921401]
                    [ 0.54509876 -0.36921401]
                    [ 0.18642675  0.87708546]
                    [ 0.18642675  0.87708546]
                    [-0.45060393 -0.208063  ]
                    [-0.45060393 -0.208063  ]]
                    S.y: [0.98021757 0.99264427 2.02575851 2.00387949 0.45185626 0.44499372
                    0.79130456 0.81487288 0.24000221 0.23988634]
                    X_shape_before: (10, 2)
                    y_size_before: 10
                    S.X: [[ 0.          1.        ]
                    [ 1.          0.        ]
                    [ 1.          1.        ]
                    [ 1.          1.        ]
                    [ 0.54509876 -0.36921401]
                    [ 0.54509876 -0.36921401]
                    [ 0.18642675  0.87708546]
                    [ 0.18642675  0.87708546]
                    [-0.45060393 -0.208063  ]
                    [-0.45060393 -0.208063  ]
                    [-0.02292587  0.0145145 ]]
                    S.y: [ 0.98021757  0.99264427  2.02575851  2.00387949  0.45185626  0.44499372
                    0.79130456  0.81487288  0.24000221  0.23988634 -0.01904616]
                    S.n_points: 1
                    S.ocba_delta: 1
                    X_shape_after: (11, 2)
                    y_size_after: 11
        """
        # OCBA (only if noise). Determination of the OCBA points depends on the
        # old X and y values.
        if self.noise and self.ocba_delta > 0 and not np.all(self.var_y > 0) and (self.mean_X.shape[0] <= 2):
            logger.warning("self.var_y <= 0. OCBA points are not generated:")
            logger.warning("There are less than 3 points or points with no variance information.")
            logger.debug("In update_design(): self.mean_X: %s", self.mean_X)
            logger.debug("In update_design(): self.var_y: %s", self.var_y)
        if self.noise and self.ocba_delta > 0 and np.all(self.var_y > 0) and (self.mean_X.shape[0] > 2):
            X_ocba = get_ocba_X(self.mean_X, self.mean_y, self.var_y, self.ocba_delta)
        else:
            X_ocba = None
        # Determine the new X0 values based on the old X and y values:
        X0 = self.get_new_X0()
        # Append OCBA points to the new design points:
        if self.noise and self.ocba_delta > 0 and np.all(self.var_y > 0):
            X0 = append(X_ocba, X0, axis=0)
        X_all = self.to_all_dim_if_needed(X0)
        logger.debug(
            "In update_design(): self.fun_control sigma and seed passed to fun(): %s %s",
            self.fun_control["sigma"],
            self.fun_control["seed"],
        )
        # (S-18): Evaluating New Solutions:
        y_mo = self.fun(X=X_all, fun_control=self.fun_control)
        # Convert multi-objective values to single-objective values:
        y0 = self._mo2so(y_mo)
        # Apply penalty for NA values works only on so values:
        y0 = apply_penalty_NA(y0, self.fun_control["penalty_NA"], verbosity=self.verbosity)
        X0, y0 = remove_nan(X0, y0, stop_on_zero_return=False)
        # Append New Solutions (only if they are not nan):
        if y0.shape[0] > 0:
            self.X = np.append(self.X, X0, axis=0)
            self.y = np.append(self.y, y0)
        else:
            # otherwise, generate a random point and append it to the design
            Xr, yr = self.generate_random_point()
            self.X = np.append(self.X, Xr, axis=0)
            self.y = np.append(self.y, yr)

    def save_result(self, filename=None, path=None, overwrite=True, verbosity=0) -> None:
        """
        Save the results to a file.
        If filename is not provided, the filename is generated based on the PREFIX using the
        `get_result_filename()` function. The results file is saved in the current working directory
        unless a path is provided. The file is saved in pickle format using the highest protocol.
        If no arguments are provided, the file is saved with the default name PREFIX + "_res.pkl".

        Args:
            filename (str):
                The filename of the results file. If not provided,
                the filename is generated based on the PREFIX using the
                `get_result_filename()` function. Default is `None`.
            path (str):
                The path to the results file. If not provided, the file
                is saved in the current working directory. Default is `None`.
            overwrite (bool):
                If `True`, the file will be overwritten if it already exists.
                Default is `True`.
            verbosity (int):
                The level of verbosity. Default is 0.

        Returns:
            None
        """
        PREFIX = self.fun_control.get("PREFIX", "result")
        if filename is None:
            filename = get_result_filename(PREFIX)
        self.save_experiment(filename=filename, path=path, overwrite=overwrite, unpickleables="file_io", verbosity=verbosity)

    def save_experiment(self, filename=None, path=None, overwrite=True, unpickleables="file_io", verbosity=0) -> None:
        """
        Save the experiment to a file.
        If no filename is provided, the filename is generated based on the PREFIX using the
        `get_experiment_filename()` function. The experiment file is saved in the current working directory
        unless a path is provided. The file is saved in pickle format using the highest protocol.
        If no arguments are provided, the file is saved with the default name PREFIX + "_exp.pkl".

        Args:
            filename (str):
                The filename of the experiment file. If not provided,
                the filename is generated based on the PREFIX using the
                `get_experiment_filename()` function. Default is `None`.
            path (str):
                The path to the experiment file. If not provided, the file
                is saved in the current working directory. Default is `None`.
            overwrite (bool):
                If `True`, the file will be overwritten if it already exists.
                Default is `True`.
            unpickleables (str):
                The type of unpickleable components to exclude. Default is "file_io", which
                excludes file I/O components like the spot_writer and logger.
                If set to any other value, the function will exclude the function, optimizer,
                surrogate, data_set, scaler, rng, and design components.
                Default is "file_io".
            verbosity (int):
                The level of verbosity. Default is 0.

        Returns:
            None
        """
        # Ensure we don't accidentally try to pickle unpicklable components
        self._close_and_del_spot_writer()
        self._remove_logger_handlers()

        S = self._get_pickle_safe_spot_tuner(unpickleables=unpickleables, verbosity=verbosity)

        # Determine the filename based on PREFIX if not provided
        PREFIX = self.fun_control.get("PREFIX", "experiment")
        if filename is None:
            filename = get_experiment_filename(PREFIX)

        if path is not None:
            filename = os.path.join(path, filename)
            if not os.path.exists(path):
                os.makedirs(path)

        # Check if the file already exists
        if filename is not None and os.path.exists(filename) and not overwrite:
            print(f"Error: File {filename} already exists. Use overwrite=True to overwrite the file.")
            return

        # Serialize the experiment dictionary to the pickle file
        if filename is not None:
            with open(filename, "wb") as handle:
                try:
                    pickle.dump(S, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f"Error during pickling: {e}")
                    raise e
            print(f"Experiment saved to {filename}")

    def _remove_logger_handlers(self) -> None:
        """
        Remove handlers from the logger to avoid pickling issues.
        """
        logger = logging.getLogger(__name__)
        for handler in list(logger.handlers):  # Copy the list to avoid modification during iteration
            logger.removeHandler(handler)

    def _close_and_del_spot_writer(self) -> None:
        """
        Delete the spot_writer attribute from the object
        if it exists and close the writer.
        """
        if hasattr(self, "spot_writer") and self.spot_writer is not None:
            self.spot_writer.flush()
            self.spot_writer.close()
            del self.spot_writer

    def _get_pickle_safe_spot_tuner(self, unpickleables="file_io", verbosity=0) -> Spot:
        """
        Create a copy of self excluding unpickleable components for safe pickling.
        This ensures no unpicklable components are passed to pickle.dump().

        Args:
            unpickleables (str):
                The type of unpickleable components to exclude. Default is "file_io", which
                excludes file I/O components like the spot_writer and logger.
                If set to any other value, the function will exclude the function, optimizer,
                surrogate, data_set, scaler, rng, and design components.
                Default is "file_io".
            verbosity (int):
                The level of verbosity. Default is 0.

        Returns:
            Spot: A copy of the Spot instance with unpickleable components removed.
        """
        # List of attributes that can't be pickled
        if unpickleables == "file_io":
            unpickleable_attrs = ["spot_writer", "logger"]
        else:
            unpickleable_attrs = ["spot_writer", "logger", "fun", "optimizer", "surrogate", "data_set", "scaler", "rng", "design"]
        # Prepare a dictionary to store picklable state
        picklable_state = {}

        # Copy picklable attributes to the dictionary
        for key, value in self.__dict__.items():
            if key not in unpickleable_attrs:
                try:
                    # Test if the attribute can be pickled
                    copy.deepcopy(value)
                    picklable_state[key] = value
                    if verbosity > 1:
                        print(f"Attribute {key} is picklable and will be included in the experiment file.")
                except Exception:
                    if verbosity > 0:
                        print(f"Attribute {key} is not picklable and will be excluded from the experiment file.")
                    continue

        # Use the dictionary to create a new instance
        picklable_instance = self.__class__.__new__(self.__class__)
        picklable_instance.__dict__.update(picklable_state)
        if verbosity > 1:
            print(f"Picklable instance created: {picklable_instance.__dict__}")

        return picklable_instance

    def _init_spot_writer(self) -> None:
        """
        Initialize the spot_writer for the current experiment if in fun_control
        the tensorboard_log is set to True
        and the spot_tensorboard_path is not None.
        Otherwise, the spot_writer is set to None.
        """
        if self.fun_control["tensorboard_log"] and self.fun_control["spot_tensorboard_path"] is not None:
            self.spot_writer = SummaryWriter(log_dir=self.fun_control["spot_tensorboard_path"])
            if self.verbosity > 0:
                print(f"_init_spot_writer(): Created spot_tensorboard_path: {self.fun_control['spot_tensorboard_path']} for SummaryWriter()")
        else:
            self.spot_writer = None
            if self.verbosity > 0:
                print("No tensorboard log created.")

    def should_continue(self, timeout_start) -> bool:
        return (self.counter < self.fun_evals) and (time.time() < timeout_start + self.max_time * 60)

    def generate_random_point(self):
        """Generate a random point in the design space.

        Returns:
            (tuple): tuple containing:
                X0 (numpy.ndarray): random point in the design space
                y0 (numpy.ndarray): function value at X

        Notes:
            If the evaluation fails, the function returns arrays of shape[0] == 0.

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import spot
                from spotpython.utils.init import fun_control_init
                fun = analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1, -1]),
                    upper = np.array([1, 1])
                    )
                S = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            )
                X0, y0 = S.generate_random_point()
                print(f"X0: {X0}")
                print(f"y0: {y0}")
                assert X0.size == 2
                assert y0.size == 1
                assert np.all(X0 >= S.lower)
                assert np.all(X0 <= S.upper)
                assert y0 >= 0
        """
        X0 = self.generate_design(
            size=1,
            repeats=1,
            lower=self.lower,
            upper=self.upper,
        )
        X0 = repair_non_numeric(X=X0, var_type=self.var_type)
        X_all = self.to_all_dim_if_needed(X0)
        logger.debug("In Spot() generate_random_point(), before calling self.fun: X_all: %s", X_all)
        logger.debug("In Spot() generate_random_point(), before calling self.fun: fun_control: %s", self.fun_control)
        # Convert multi-objective values to single-objective values
        # TODO: Store y_mo in self.y_mo (append new values)
        y_mo = self.fun(X=X_all, fun_control=self.fun_control)
        y0 = self._mo2so(y_mo)
        # Apply penalty for NA values works only on so values:
        y0 = apply_penalty_NA(y0, self.fun_control["penalty_NA"], verbosity=self.verbosity)
        X0, y0 = remove_nan(X0, y0, stop_on_zero_return=False)
        return X0, y0

    def show_progress_if_needed(self, timeout_start) -> None:
        """Show progress bar if `show_progress` is `True`. If
        self.progress_file is not `None`, the progress bar is saved
        in the file with the name `self.progress_file`.

        Args:
            self (object): Spot object
            timeout_start (float): start time

        Returns:
            (NoneType): None
        """
        if not self.show_progress:
            return
        if isfinite(self.fun_evals):
            progress_bar(progress=self.counter / self.fun_evals, y=self.min_y, filename=self.progress_file)
        else:
            progress_bar(progress=(time.time() - timeout_start) / (self.max_time * 60), y=self.min_y, filename=self.progress_file)

    def generate_design(self, size, repeats, lower, upper) -> np.array:
        """Generate a design with `size` points in the interval [lower, upper].

        Args:
            size (int): number of points
            repeats (int): number of repeats
            lower (numpy.ndarray): lower bound of the design space
            upper (numpy.ndarray): upper bound of the design space

        Returns:
            (numpy.ndarray): design points

        Examples:
            >>> import numpy as np
                from spotpython.spot import spot
                from spotpython.utils.init import design_control_init
                from spotpython.fun.objectivefunctions import Analytical
                design_control = design_control_init(init_size=3)
                fun_control = fun_control_init(
                    lower = np.array([-1, -1]),
                    upper = np.array([1, 1]),
                    fun_evals=fun_evals,
                    tolerance_x = np.sqrt(np.spacing(1))
                    )
                S = spot.Spot(fun = analytical().fun_sphere,
                            fun_control = fun_control,
                            design_control = design_control)
                X = S.generate_design(size=3, repeats=1, lower=np.array([0, 0]), upper=np.array([100, 1]))
                assert X.shape[0] == 3
                assert X.shape[1] == 2
                print(X)
                    array([[77.25493789,  0.31539299],
                    [59.32133757,  0.93854273],
                    [27.4698033 ,  0.3959685 ]])
        """
        return self.design.scipy_lhd(n=size, repeats=repeats, lower=lower, upper=upper)

    def update_writer(self) -> None:
        if hasattr(self, "spot_writer") and self.spot_writer is not None:
            # get the last y value:
            y_last = self.y[-1].copy()
            if self.noise is False:
                y_min = self.min_y.copy()
                X_min = self.min_X.copy()
                # y_min: best y value so far
                # y_last: last y value, can be worse than y_min
                self.spot_writer.add_scalars("spot_y", {"min": y_min, "last": y_last}, self.counter)
                # X_min: X value of the best y value so far
                self.spot_writer.add_scalars("spot_X", {f"X_{i}": X_min[i] for i in range(self.k)}, self.counter)
            else:
                # get the last n y values:
                y_last_n = self.y[-self.fun_repeats :].copy()
                # y_min_mean: best mean y value so far
                y_min_mean = self.min_mean_y.copy()
                # X_min_mean: X value of the best mean y value so far
                X_min_mean = self.min_mean_X.copy()
                # y_min_var: variance of the min y value so far
                y_min_var = self.min_var_y.copy()
                self.spot_writer.add_scalar("spot_y_min_var", y_min_var, self.counter)
                # y_min_mean: best mean y value so far (see above)
                self.spot_writer.add_scalar("spot_y", y_min_mean, self.counter)
                # last n y values (noisy):
                self.spot_writer.add_scalars("spot_y", {f"y_last_n{i}": y_last_n[i] for i in range(self.fun_repeats)}, self.counter)
                # X_min_mean: X value of the best mean y value so far (see above)
                self.spot_writer.add_scalars("spot_X_noise", {f"X_min_mean{i}": X_min_mean[i] for i in range(self.k)}, self.counter)
            # get last value of self.X and convert to dict. take the values from self.var_name as keys:
            X_last = self.X[-1].copy()
            config = {self.var_name[i]: X_last[i] for i in range(self.k)}
            # var_dict = assign_values(X, get_var_name(fun_control))
            # config = list(generate_one_config_from_var_dict(var_dict, fun_control))[0]
            # hyperparameters X and value y of the last configuration:
            # see: https://github.com/pytorch/pytorch/issues/32651
            # self.spot_writer.add_hparams(config, {"spot_y": y_last}, run_name=self.spot_tensorboard_path)
            self.spot_writer.add_hparams(config, {"hp_metric": y_last})
            self.spot_writer.flush()
            if self.verbosity > 0:
                print("update_writer(): Done.")
        else:
            if self.verbosity > 0:
                print("No spot_writer available.")

    def suggest_new_X(self) -> np.array:
        """
        Compute `n_points` new infill points in natural units.
        These diffrent points are computed by the optimizer using increasing seed.
        The optimizer searches in the ranges from `lower_j` to `upper_j`.
        The method `infill()` is used as the objective function.

        Returns:
            (numpy.ndarray): `n_points` infill points in natural units, each of dim k

        Note:
            This is step (S-14a) in [bart21i].

        Examples:
            >>> import numpy as np
                from spotpython.spot import Spot
                from spotpython.fun import Analytical
                from spotpython.utils.init import fun_control_init
                nn = 3
                fun_sphere = Analytical().fun_sphere
                fun_control = fun_control_init(
                        lower = np.array([-1, -1]),
                        upper = np.array([1, 1]),
                        n_points=nn,
                        )
                S = Spot(
                    fun=fun_sphere,
                    fun_control=fun_control,
                    )
                S.X = S.design.scipy_lhd(
                    S.design_control["init_size"], lower=S.lower, upper=S.upper
                )
                print(f"S.X: {S.X}")
                S.y = S.fun(S.X)
                print(f"S.y: {S.y}")
                S.fit_surrogate()
                X0 = S.suggest_new_X()
                print(f"X0: {X0}")
                assert X0.size == S.n_points * S.k
                assert X0.ndim == 2
                assert X0.shape[0] == nn
                assert X0.shape[1] == 2
                spot_1.X: [[ 0.86352963  0.7892358 ]
                            [-0.24407197 -0.83687436]
                            [ 0.36481882  0.8375811 ]
                            [ 0.415331    0.54468512]
                            [-0.56395091 -0.77797854]
                            [-0.90259409 -0.04899292]
                            [-0.16484832  0.35724741]
                            [ 0.05170659  0.07401196]
                            [-0.78548145 -0.44638164]
                            [ 0.64017497 -0.30363301]]
                spot_1.y: [1.36857656 0.75992983 0.83463487 0.46918172 0.92329124 0.8170764
                0.15480068 0.00815134 0.81623768 0.502017  ]
                X0: [[0.00154544 0.003962  ]
                    [0.00165526 0.00410847]
                    [0.00165685 0.0039177 ]]
        """
        # (S-14a) Optimization on the surrogate:
        new_X = np.zeros([self.n_points, self.k], dtype=float)
        optimizer_name = self.optimizer.__name__
        optimizers = {
            "dual_annealing": lambda: self.optimizer(func=self.infill, bounds=self.de_bounds),
            "differential_evolution": lambda: self.optimizer(
                func=self.infill,
                bounds=self.de_bounds,
                maxiter=self.optimizer_control["max_iter"],
                seed=self.optimizer_control["seed"],
            ),
            "direct": lambda: self.optimizer(func=self.infill, bounds=self.de_bounds, eps=1e-2),
            "shgo": lambda: self.optimizer(func=self.infill, bounds=self.de_bounds),
            "basinhopping": lambda: self.optimizer(func=self.infill, x0=self.min_X, minimizer_kwargs={"method": "Nelder-Mead"}),
            "default": lambda: self.optimizer(func=self.infill, bounds=self.de_bounds),
        }
        for i in range(self.n_points):
            self.optimizer_control["seed"] = self.optimizer_control["seed"] + i
            result = optimizers.get(optimizer_name, optimizers["default"])()
            new_X[i][:] = result.x
        return np.unique(new_X, axis=0)

    def infill(self, x) -> float:
        """
        Infill (acquisition) function. Evaluates one point on the surrogate via `surrogate.predict(x.reshape(1,-1))`,
        if `sklearn` surrogates are used or `surrogate.predict(x.reshape(1,-1), return_val=self.infill_criterion)`
        if the internal surrogate `kriging` is selected.
        This method is passed to the optimizer in `suggest_new_X`, i.e., the optimizer is called via
        `self.optimizer(func=self.infill)`.

        Args:
            x (array): point in natural units with shape `(1, dim)`.

        Returns:
            (numpy.ndarray): value based on infill criterion, e.g., `"ei"`. Shape `(1,)`.
                The objective function value `y` that is used as a base value for the
                infill criterion is calculated in natural units.

        Note:
            This is step (S-12) in [bart21i].
        """
        # Reshape x to have shape (1, -1) because the predict method expects a 2D array
        X = x.reshape(1, -1)
        if isinstance(self.surrogate, Kriging) and getattr(self.surrogate, "name", None) == "kriging":
            return self.surrogate.predict(X, return_val=self.infill_criterion)
        else:
            return self.surrogate.predict(X)

    def plot_progress(self, show=True, log_x=False, log_y=False, filename="plot.png", style=["ko", "k", "ro-"], dpi=300, tkagg=False) -> None:
        """Plot the progress of the hyperparameter tuning (optimization).

        Args:
            show (bool):
                Show the plot.
            log_x (bool):
                Use logarithmic scale for x-axis.
            log_y (bool):
                Use logarithmic scale for y-axis.
            filename (str):
                Filename to save the plot.
            style (list):
                Style of the plot. Default: ['k', 'ro-'], i.e., the initial points are plotted as a black line
                and the subsequent points as red dots connected by a line.

        Returns:
            None

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import spot
                from spotpython.utils.init import (
                    fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
                    )
                # number of initial points:
                ni = 7
                # number of points
                fun_evals = 10
                fun = analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1, -1]),
                    upper = np.array([1, 1])
                    fun_evals=fun_evals,
                    tolerance_x = np.sqrt(np.spacing(1))
                    )
                design_control=design_control_init(init_size=ni)
                surrogate_control=surrogate_control_init(n_theta=3)
                S = spot.Spot(fun=fun,
                            fun_control=fun_control
                            design_control=design_control,
                            surrogate_control=surrogate_control,)
                S.run()
                S.plot_progress(log_y=True)

        """
        if tkagg:
            matplotlib.use("TkAgg")
        fig = pylab.figure(figsize=(9, 6))
        s_y = pd.Series(self.y)
        s_c = s_y.cummin()
        n_init = self.design_control["init_size"] * self.design_control["repeats"]

        ax = fig.add_subplot(211)
        if n_init <= len(s_y):
            ax.plot(
                range(1, n_init + 1),
                s_y[:n_init],
                style[0],
                range(1, n_init + 2),
                [s_c[:n_init].min()] * (n_init + 1),
                style[1],
                range(n_init + 1, len(s_c) + 1),
                s_c[n_init:],
                style[2],
            )
        else:
            # plot only s_y values:
            ax.plot(range(1, len(s_y) + 1), s_y, style[0])
            logger.warning("Less evaluations ({len(s_y)}) than initial design points ({n_init}).")
        ax.set_xlabel("Iteration")
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        if filename is not None:
            pylab.savefig(filename, dpi=dpi, bbox_inches="tight")
        if show:
            pylab.show()

    def plot_model(self, y_min=None, y_max=None) -> None:
        """
        Plot the model fit for 1-dim objective functions.

        Args:
            self (object):
                spot object
            y_min (float, optional):
                y range, lower bound.
            y_max (float, optional):
                y range, upper bound.

        Returns:
            None

        Examples:
            >>> import numpy as np
                from spotpython.utils.init import (
                    fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
                    )
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import spot
                # number of initial points:
                ni = 3
                # number of points
                fun_evals = 7
                fun = analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1]),
                    upper = np.array([1]),
                    fun_evals=fun_evals,
                    tolerance_x = np.sqrt(np.spacing(1))
                    )
                design_control=design_control_init(init_size=ni)

                S = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control
                S.run()
                S.plot_model()
        """
        if self.k == 1:
            X_test = np.linspace(self.lower[0], self.upper[0], 100)
            y_mo = self.fun(X=X_test.reshape(-1, 1), fun_control=self.fun_control)
            # convert multi-objective values to single-objective values
            y_test = self._mo2so(y_mo)
            # Apply penalty for NA values works only on so values:
            y_test = apply_penalty_NA(y_test, self.fun_control["penalty_NA"], verbosity=self.verbosity)
            if isinstance(self.surrogate, Kriging) and getattr(self.surrogate, "name", None) == "kriging":
                y_hat = self.surrogate.predict(X_test[:, np.newaxis], return_val="y")
            else:
                y_hat = self.surrogate.predict(X_test[:, np.newaxis])
            plt.plot(X_test, y_hat, label="Model")
            plt.plot(X_test, y_test, label="True function")
            plt.scatter(self.X, self.y, edgecolor="b", s=20, label="Samples")
            plt.scatter(self.X[-1], self.y[-1], edgecolor="r", s=30, label="Last Sample")
            if self.noise:
                plt.scatter(self.min_mean_X, self.min_mean_y, edgecolor="g", s=30, label="Best Sample (mean)")
            else:
                plt.scatter(self.min_X, self.min_y, edgecolor="g", s=30, label="Best Sample")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xlim((self.lower[0], self.upper[0]))
            if y_min is None:
                y_min = min([min(self.y), min(y_test)])
            if y_max is None:
                y_max = max([max(self.y), max(y_test)])
            plt.ylim((y_min, y_max))
            plt.legend(loc="best")
            # plt.title(self.surrogate.__class__.__name__ + ". " + str(self.counter) + ": " + str(self.min_y))
            if self.noise:
                plt.title("fun_evals: " + str(self.counter) + ". min_y (noise): " + str(np.round(self.min_y, 6)) + " min_mean_y: " + str(np.round(self.min_mean_y, 6)))
            else:
                plt.title("fun_evals: " + str(self.counter) + ". min_y: " + str(np.round(self.min_y, 6)))
            plt.show()

    def print_results(self, print_screen=True, dict=None) -> list[str]:
        """Print results from the run:
            1. min y
            2. min X
            If `noise == True`, additionally the following values are printed:
            3. min mean y
            4. min mean X

        Args:
            print_screen (bool, optional):
                print results to screen

        Returns:
            output (list):
                list of results
        """
        output = []
        if print_screen:
            print(f"min y: {self.min_y}")
            if self.noise:
                print(f"min mean y: {self.min_mean_y}")
        if self.noise:
            res = self.to_all_dim(self.min_mean_X.reshape(1, -1))
        else:
            res = self.to_all_dim(self.min_X.reshape(1, -1))
        for i in range(res.shape[1]):
            if self.all_var_name is None:
                var_name = "x" + str(i)
            else:
                var_name = self.all_var_name[i]
                var_type = self.all_var_type[i]
                if var_type == "factor" and dict is not None:
                    val = get_ith_hyperparameter_name_from_fun_control(fun_control=dict, key=var_name, i=int(res[0][i]))
                else:
                    val = res[0][i]
            if print_screen:
                print(var_name + ":", val)
            output.append([var_name, val])
        return output

    def get_tuned_hyperparameters(self, fun_control=None) -> dict:
        """Return the tuned hyperparameter values from the run.
        If `noise == True`, the mean values are returned.

        Args:
            fun_control (dict, optional):
                fun_control dictionary

        Returns:
            (dict): dictionary of tuned hyperparameters.

        Examples:
            >>> from spotpython.utils.device import getDevice
                from math import inf
                from spotpython.utils.init import fun_control_init
                import numpy as np
                from spotpython.hyperparameters.values import set_control_key_value
                from spotpython.data.diabetes import Diabetes
                MAX_TIME = 1
                FUN_EVALS = 10
                INIT_SIZE = 5
                WORKERS = 0
                PREFIX="037"
                DEVICE = getDevice()
                DEVICES = 1
                TEST_SIZE = 0.4
                TORCH_METRIC = "mean_squared_error"
                dataset = Diabetes()
                fun_control = fun_control_init(
                    _L_in=10,
                    _L_out=1,
                    _torchmetric=TORCH_METRIC,
                    PREFIX=PREFIX,
                    TENSORBOARD_CLEAN=True,
                    data_set=dataset,
                    device=DEVICE,
                    enable_progress_bar=False,
                    fun_evals=FUN_EVALS,
                    log_level=50,
                    max_time=MAX_TIME,
                    num_workers=WORKERS,
                    show_progress=True,
                    test_size=TEST_SIZE,
                    tolerance_x=np.sqrt(np.spacing(1)),
                    )
                from spotpython.light.regression.netlightregression import NetLightRegression
                from spotpython.hyperdict.light_hyper_dict import LightHyperDict
                from spotpython.hyperparameters.values import add_core_model_to_fun_control
                add_core_model_to_fun_control(fun_control=fun_control,
                                            core_model=NetLightRegression,
                                            hyper_dict=LightHyperDict)
                from spotpython.hyperparameters.values import set_control_hyperparameter_value
                set_control_hyperparameter_value(fun_control, "l1", [7, 8])
                set_control_hyperparameter_value(fun_control, "epochs", [3, 5])
                set_control_hyperparameter_value(fun_control, "batch_size", [4, 5])
                set_control_hyperparameter_value(fun_control, "optimizer", [
                                "Adam",
                                "RAdam",
                            ])
                set_control_hyperparameter_value(fun_control, "dropout_prob", [0.01, 0.1])
                set_control_hyperparameter_value(fun_control, "lr_mult", [0.5, 5.0])
                set_control_hyperparameter_value(fun_control, "patience", [2, 3])
                set_control_hyperparameter_value(fun_control, "act_fn",[
                                "ReLU",
                                "LeakyReLU"
                            ] )
                from spotpython.utils.init import design_control_init, surrogate_control_init
                design_control = design_control_init(init_size=INIT_SIZE)
                surrogate_control = surrogate_control_init(method="regression",
                                                            n_theta=2)
                from spotpython.fun.hyperlight import HyperLight
                fun = HyperLight(log_level=50).fun
                from spotpython.spot import spot
                spot_tuner = spot.Spot(fun=fun,
                                    fun_control=fun_control,
                                    design_control=design_control,
                                    surrogate_control=surrogate_control)
                spot_tuner.run()
                spot_tuner.get_tuned_hyperparameters()
                    {'l1': 7.0,
                    'epochs': 5.0,
                    'batch_size': 4.0,
                    'act_fn': 0.0,
                    'optimizer': 0.0,
                    'dropout_prob': 0.01,
                    'lr_mult': 5.0,
                    'patience': 3.0,
                    'initialization': 1.0}

        """
        output = []
        if self.noise:
            res = self.to_all_dim(self.min_mean_X.reshape(1, -1))
        else:
            res = self.to_all_dim(self.min_X.reshape(1, -1))
        for i in range(res.shape[1]):
            if self.all_var_name is None:
                var_name = "x" + str(i)
            else:
                var_name = self.all_var_name[i]
                var_type = self.all_var_type[i]
                if var_type == "factor" and fun_control is not None:
                    val = get_ith_hyperparameter_name_from_fun_control(fun_control=fun_control, key=var_name, i=int(res[0][i]))
                else:
                    val = res[0][i]
            output.append([var_name, val])
        # convert list to a dictionary
        output = dict(output)
        return output

    def chg(self, x, y, z0, i, j) -> list:
        """
        Change the values of elements at indices `i` and `j` in the array `z0` to `x` and `y`, respectively.

        Args:
            x (int or float):
                The new value for the element at index `i`.
            y (int or float):
                The new value for the element at index `j`.
            z0 (list or numpy.ndarray):
                The array to be modified.
            i (int):
                The index of the element to be changed to `x`.
            j (int):
                The index of the element to be changed to `y`.

        Returns:
            (list) or (numpy.ndarray): The modified array.

        Examples:
                >>> import numpy as np
                    from spotpython.fun.objectivefunctions import Analytical
                    from spotpython.spot import spot
                    from spotpython.utils.init import (
                        fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
                    )
                    fun = analytical().fun_sphere
                    fun_control = fun_control_init(
                        lower = np.array([-1]),
                        upper = np.array([1]),
                    )
                    S = spot.Spot(fun=fun,
                                func_control=fun_control)
                    z0 = [1, 2, 3]
                    print(f"Before: {z0}")
                    new_val_1 = 4
                    new_val_2 = 5
                    index_1 = 0
                    index_2 = 2
                    S.chg(x=new_val_1, y=new_val_2, z0=z0, i=index_1, j=index_2)
                    print(f"After: {z0}")
                    Before: [1, 2, 3]
                    After: [4, 2, 5]
        """
        z0[i] = x
        z0[j] = y
        return z0

    def process_z00(self, z00, use_min=True) -> list:
        """Process each entry in the `z00` array according to the corresponding type
        in the `self.var_type` list.
        Specifically, if the type is "float", the function will calculate the mean of the two `z00` values.
        If the type is not "float", the function will retrun the maximum of the two `z00` values.

        Args:
            z00 (numpy.ndarray):
                Array of values to process.
            use_min (bool):
                If `True`, the minimum value is returned. If `False`, the maximum value is returned.

        Returns:
            (list): Processed values.

        Examples:
            from spotpython.spot import spot
            import numpy as np
            import random
            z00 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
            spot.var_type = ["float", "int", "int", "float"]
            spot.process_z00(z00)
            [3, 6, 7, 6]

        """
        result = []
        for i in range(len(self.var_type)):
            if self.var_type[i] == "float":
                mean_value = np.mean(z00[:, i])
                result.append(mean_value)
            else:  # var_type[i] == 'int'
                if use_min:
                    min_value = min(z00[:, i])
                    result.append(min_value)
                else:
                    max_value = max(z00[:, i])
                    result.append(max_value)
        return result

    def plot_contour(
        self,
        i=0,
        j=1,
        min_z=None,
        max_z=None,
        show=True,
        filename=None,
        n_grid=50,
        contour_levels=10,
        dpi=200,
        title="",
        figsize=(12, 6),
        use_min=False,
        use_max=True,
        tkagg=False,
    ) -> None:
        """Plot the contour of any dimension."""

        def generate_mesh_grid(lower, upper, grid_points):
            """Generate a mesh grid for the given range."""
            x = np.linspace(lower[i], upper[i], num=grid_points)
            y = np.linspace(lower[j], upper[j], num=grid_points)
            return np.meshgrid(x, y), x, y

        def validate_types(var_type, lower, upper):
            """Validate if the dimensions of var_type, lower, and upper are the same."""
            if var_type is not None:
                if len(var_type) != len(lower) or len(var_type) != len(upper):
                    raise ValueError("The dimensions of var_type, lower, and upper must be the same.")

        def setup_plot():
            """Setup the plot with specified figure size."""
            fig = pylab.figure(figsize=figsize)
            return fig

        def predict_contour_values(X, Y, z0):
            """Predict contour values based on the surrogate model."""
            grid_points = np.c_[np.ravel(X), np.ravel(Y)]
            predictions = []

            for x, y in grid_points:
                adjusted_z0 = self.chg(x, y, z0.copy(), i, j)
                prediction = self.surrogate.predict(np.array([adjusted_z0]))
                predictions.append(prediction[0])

            Z = np.array(predictions).reshape(X.shape)
            return Z

        def plot_contour_subplots(X, Y, Z, ax, min_z, max_z, contour_levels):
            """Plot the contour and 3D surface subplots."""
            contour = ax.contourf(X, Y, Z, contour_levels, zorder=1, cmap="jet", vmin=min_z, vmax=max_z)
            pylab.colorbar(contour, ax=ax)

        if tkagg:
            matplotlib.use("TkAgg")
        fig = setup_plot()

        (X, Y), x, y = generate_mesh_grid(self.lower, self.upper, n_grid)
        validate_types(self.var_type, self.lower, self.upper)

        z00 = np.array([self.lower, self.upper])
        Z_list, X_list, Y_list = [], [], []

        if use_min:
            z0_min = self.process_z00(z00, use_min=True)
            Z_min = predict_contour_values(X, Y, z0_min)
            Z_list.append(Z_min)
            X_list.append(X)
            Y_list.append(Y)

        if use_max:
            z0_max = self.process_z00(z00, use_min=False)
            Z_max = predict_contour_values(X, Y, z0_max)
            Z_list.append(Z_max)
            X_list.append(X)
            Y_list.append(Y)

        if Z_list:  # Ensure that there is at least one Z to stack
            Z_combined = np.vstack(Z_list)
            X_combined = np.vstack(X_list)
            Y_combined = np.vstack(Y_list)

        if min_z is None:
            min_z = np.min(Z_combined)
        if max_z is None:
            max_z = np.max(Z_combined)

        ax_contour = fig.add_subplot(221)
        plot_contour_subplots(X_combined, Y_combined, Z_combined, ax_contour, min_z, max_z, contour_levels)

        if self.var_name is None:
            ax_contour.set_xlabel(f"x{i}")
            ax_contour.set_ylabel(f"x{j}")
        else:
            ax_contour.set_xlabel(f"x{i}: {self.var_name[i]}")
            ax_contour.set_ylabel(f"x{j}: {self.var_name[j]}")

        ax_3d = fig.add_subplot(222, projection="3d")
        ax_3d.plot_surface(X_combined, Y_combined, Z_combined, rstride=3, cstride=3, alpha=0.9, cmap="jet", vmin=min_z, vmax=max_z)

        if self.var_name is None:
            ax_3d.set_xlabel(f"x{i}")
            ax_3d.set_ylabel(f"x{j}")
        else:
            ax_3d.set_xlabel(f"x{i}: {self.var_name[i]}")
            ax_3d.set_ylabel(f"x{j}: {self.var_name[j]}")

        plt.title(title)

        if filename:
            pylab.savefig(filename, bbox_inches="tight", dpi=dpi, pad_inches=0)

        if show:
            pylab.show()

    def plot_important_hyperparameter_contour(
        self,
        threshold=0.0,
        filename=None,
        show=True,
        max_imp=None,
        title="",
        scale_global=False,
        n_grid=50,
        contour_levels=10,
        dpi=200,
        use_min=False,
        use_max=True,
        tkagg=False,
    ) -> None:
        """
        Plot the contour of important hyperparameters.
        Calls `plot_contour` for each pair of important hyperparameters.
        Importance can be specified by the threshold.

        Args:
            threshold (float):
                threshold for the importance. Not used any more in spotpython >= 0.13.2.
            filename (str):
                filename of the plot
            show (bool):
                show the plot. Default is `True`.
            max_imp (int):
                maximum number of important hyperparameters. If there are more important hyperparameters
                than `max_imp`, only the max_imp important ones are selected.
            title (str):
                title of the plots
            scale_global (bool):
                scale the z-axis globally. Default is `False`.
            n_grid (int):
                number of grid points. Default is 50.
            contour_levels (int):
                number of contour levels. Default is 10.
            dpi (int):
                dpi of the plot. Default is 200.
            use_min (bool):
                Use the minimum value for determing the hidden dimensions in the plot for categorical and
                integer parameters.
                In 3d-plots, only two variables can be independent. The remaining input variables are set
                to their minimum value.
                Default is `False`.
                If use_min and use_max are both `True`, both values are used.
            use_max (bool):
                Use the minimum value for determing the hidden dimensions in the plot for categorical and
                integer parameters.
                In 3d-plots, only two variables can be independent. The remaining input variables are set
                to their minimum value.
                Default is `True`.
                If use_min and use_max are both `True`, both values are used.

        Returns:
            None.

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import spot
                from spotpython.utils.init import (
                    fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
                    )
                # number of initial points:
                ni = 5
                # number of points
                fun_evals = 10
                fun = analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1, -1, -1]),
                    upper = np.array([1, 1, 1]),
                    fun_evals=fun_evals,
                    tolerance_x = np.sqrt(np.spacing(1))
                    )
                design_control=design_control_init(init_size=ni)
                surrogate_control=surrogate_control_init(n_theta=3)
                S = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control,
                            surrogate_control=surrogate_control,)
                S.run()
                S.plot_important_hyperparameter_contour()

        """
        impo = self.print_importance(threshold=threshold, print_screen=True)
        indices = sort_by_kth_and_return_indices(array=impo, k=1)
        # take the first max_imp values from the indices array
        if max_imp is not None:
            indices = indices[:max_imp]
        if scale_global:
            min_z = min(self.y)
            max_z = max(self.y)
        else:
            min_z = None
            max_z = None
        for i in indices:
            for j in indices:
                if j > i:
                    if filename is not None:
                        filename_full = filename + "_contour_" + str(i) + "_" + str(j) + ".png"
                    else:
                        filename_full = None
                    self.plot_contour(
                        i=i,
                        j=j,
                        min_z=min_z,
                        max_z=max_z,
                        filename=filename_full,
                        show=show,
                        title=title,
                        n_grid=n_grid,
                        contour_levels=contour_levels,
                        dpi=dpi,
                        use_max=use_max,
                        use_min=use_min,
                        tkagg=tkagg,
                    )

    def get_importance(self) -> list:
        """Get importance of each variable and return the results as a list.

        Returns:
            output (list):
                list of results. If the surrogate has more than one theta values,
                the importance is calculated. Otherwise, a list of zeros is returned.

        """
        if self.surrogate.n_theta > 1 and self.var_name is not None:
            output = [0] * len(self.all_var_name)
            theta = np.power(10, self.surrogate.theta)
            imp = 100 * theta / np.max(theta)
            ind = find_indices(A=self.var_name, B=self.all_var_name)
            j = 0
            for i in ind:
                output[i] = imp[j]
                j = j + 1
            return output
        else:
            print("Importance requires more than one theta values (n_theta>1).")
            # return a list of zeros of length len(all_var_name)
            return [0] * len(self.all_var_name)

    def print_importance(self, threshold=0.1, print_screen=True) -> list:
        """Print importance of each variable and return the results as a list.

        Args:
            threshold (float):
                threshold for printing
            print_screen (boolean):
                if `True`, values are also printed on the screen. Default is `True`.

        Returns:
            output (list):
                list of results
        """
        output = []
        if self.surrogate.n_theta > 1:
            theta = np.power(10, self.surrogate.theta)
            imp = 100 * theta / np.max(theta)
            # imp = imp[imp >= threshold]
            if self.var_name is None:
                for i in range(len(imp)):
                    if imp[i] >= threshold:
                        if print_screen:
                            print("x", i, ": ", imp[i])
                        output.append("x" + str(i) + ": " + str(imp[i]))
            else:
                var_name = [self.var_name[i] for i in range(len(imp))]
                for i in range(len(imp)):
                    if imp[i] >= threshold:
                        if print_screen:
                            print(var_name[i] + ": ", imp[i])
                    output.append([var_name[i], imp[i]])
        else:
            print("Importance requires more than one theta values (n_theta>1).")
        return output

    def plot_importance(self, threshold=0.1, filename=None, dpi=300, show=True, tkagg=False) -> None:
        """Plot the importance of each variable.

        Args:
            threshold (float):
                The threshold of the importance.
            filename (str):
                The filename of the plot.
            dpi (int):
                The dpi of the plot.
            show (bool):
                Show the plot. Default is `True`.

        Returns:
            None
        """
        if self.surrogate.n_theta > 1:
            if tkagg:
                matplotlib.use("TkAgg")
            theta = np.power(10, self.surrogate.theta)
            imp = 100 * theta / np.max(theta)
            idx = np.where(imp > threshold)[0]
            if self.var_name is None:
                plt.bar(range(len(imp[idx])), imp[idx])
                plt.xticks(range(len(imp[idx])), ["x" + str(i) for i in idx])
            else:
                var_name = [self.var_name[i] for i in idx]
                plt.bar(range(len(imp[idx])), imp[idx])
                plt.xticks(range(len(imp[idx])), var_name)
            if filename is not None:
                plt.savefig(filename, bbox_inches="tight", dpi=dpi)
            if show:
                plt.show()

    def parallel_plot(self, show=False) -> go.Figure:
        """
        Parallel plot.

        Args:
            self (object):
                Spot object
            show (bool):
                show the plot. Default is `False`.

        Returns:
                fig (plotly.graph_objects.Figure): figure object

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import spot
                from spotpython.utils.init import (
                    fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
                    )
                # number of initial points:
                ni = 5
                # number of points
                fun_evals = 10
                fun = analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1, -1, -1]),
                    upper = np.array([1, 1, 1]),
                    fun_evals=fun_evals,
                    tolerance_x = np.sqrt(np.spacing(1))
                    )
                design_control=design_control_init(init_size=ni)
                surrogate_control=surrogate_control_init(n_theta=3)
                S = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control,
                            surrogate_control=surrogate_control,)
                S.run()
                S.parallel_plot()

        """
        X = self.X
        y = self.y
        df = pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1), columns=self.var_name + ["y"])

        fig = go.Figure(
            data=go.Parcoords(
                line=dict(color=df["y"], colorscale="Jet", showscale=True, cmin=min(df["y"]), cmax=max(df["y"])),
                dimensions=list([dict(range=[min(df.iloc[:, i]), max(df.iloc[:, i])], label=df.columns[i], values=df.iloc[:, i]) for i in range(len(df.columns) - 1)]),
            )
        )
        if show:
            fig.show()
        return fig

    def _reattach_logger_handlers(self) -> None:
        """
        Reattach handlers to the logger after unpickling.
        """
        logger = logging.getLogger(__name__)
        # configure the handler and formatter as needed
        py_handler = logging.FileHandler(f"{__name__}.log", mode="w")
        py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
        # add formatter to the handler
        py_handler.setFormatter(py_formatter)
        # add handler to the logger
        logger.addHandler(py_handler)

    def _de_serialize_dicts(self) -> tuple:
        """
        Deserialize the spot object and return the dictionaries.

        Args:
            self (object):
                Spot object

        Returns:
            (tuple):
                tuple containing dictionaries of spot object:
                fun_control (dict): function control dictionary,
                design_control (dict): design control dictionary,
                optimizer_control (dict): optimizer control dictionary,
                spot_tuner_control (dict): spot tuner control dictionary, and
                surrogate_control (dict): surrogate control dictionary
        """
        spot_tuner = copy.deepcopy(self)
        spot_tuner_control = vars(spot_tuner)

        fun_control = copy.deepcopy(spot_tuner_control["fun_control"])
        design_control = copy.deepcopy(spot_tuner_control["design_control"])
        optimizer_control = copy.deepcopy(spot_tuner_control["optimizer_control"])
        surrogate_control = copy.deepcopy(spot_tuner_control["surrogate_control"])

        # remove keys from the dictionaries:
        spot_tuner_control.pop("fun_control", None)
        spot_tuner_control.pop("design_control", None)
        spot_tuner_control.pop("optimizer_control", None)
        spot_tuner_control.pop("surrogate_control", None)
        spot_tuner_control.pop("spot_writer", None)
        spot_tuner_control.pop("design", None)
        spot_tuner_control.pop("fun", None)
        spot_tuner_control.pop("optimizer", None)
        spot_tuner_control.pop("rng", None)
        spot_tuner_control.pop("surrogate", None)

        fun_control.pop("core_model", None)
        fun_control.pop("metric_river", None)
        fun_control.pop("metric_sklearn", None)
        fun_control.pop("metric_torch", None)
        fun_control.pop("prep_model", None)
        fun_control.pop("spot_writer", None)
        fun_control.pop("test", None)
        fun_control.pop("train", None)

        surrogate_control.pop("model_optimizer", None)
        surrogate_control.pop("surrogate", None)

        return (fun_control, design_control, optimizer_control, spot_tuner_control, surrogate_control)

    def _write_db_dict(self) -> None:
        """Writes a dictionary with the experiment parameters to the json file spotpython_db.json.

        Args:
            self (object): Spot object

        Returns:
            (NoneType): None

        """
        # get the time in seconds from 1.1.1970 and convert the time to a string
        t_str = str(time.time())
        ident = str(self.fun_control["PREFIX"]) + "_" + t_str

        (
            fun_control,
            design_control,
            optimizer_control,
            spot_tuner_control,
            surrogate_control,
        ) = self._de_serialize_dicts()
        print("\n**")
        print("The following dictionaries are written to the json file spotpython_db.json:")
        print("fun_control:")
        pprint.pprint(fun_control)

        # Iterate over a list of the keys to avoid modifying the dictionary during iteration
        for key in list(fun_control.keys()):
            if not isinstance(fun_control[key], (int, float, str, list, dict)):
                # remove the key from the dictionary
                print(f"Removing non-serializable key: {key}")
                fun_control.pop(key)

        print("fun_control after removing non-serializabel keys:")
        pprint.pprint(fun_control)
        pprint.pprint(fun_control)
        print("design_control:")
        pprint.pprint(design_control)
        print("optimizer_control:")
        pprint.pprint(optimizer_control)
        print("spot_tuner_control:")
        pprint.pprint(spot_tuner_control)
        print("surrogate_control:")
        pprint.pprint(surrogate_control)
        #
        # Generate a description of the results:
        # if spot_tuner_control['min_y'] exists:
        try:
            result = f"""
                      Results for {ident}: Finally, the best value is {spot_tuner_control['min_y']}
                      at {spot_tuner_control['min_X']}."""
            #
            db_dict = {
                "data": {
                    "id": str(ident),
                    "result": result,
                    "fun_control": fun_control,
                    "design_control": design_control,
                    "surrogate_control": surrogate_control,
                    "optimizer_control": optimizer_control,
                    "spot_tuner_control": spot_tuner_control,
                }
            }
            # Check if the directory "db_dicts" exists.
            if not os.path.exists("db_dicts"):
                try:
                    os.makedirs("db_dicts")
                except OSError as e:
                    raise Exception(f"Error creating directory: {e}")

            if os.path.exists("db_dicts"):
                try:
                    # Open the file in append mode to add each new dict as a new line
                    with open("db_dicts/" + self.fun_control["db_dict_name"], "a") as f:
                        # Using json.dumps to convert the dict to a JSON formatted string
                        # We then write this string to the file followed by a newline character
                        # This ensures that each dict is on its own line, conforming to the JSON Lines format
                        f.write(json.dumps(db_dict, cls=NumpyEncoder) + "\n")
                except OSError as e:
                    raise Exception(f"Error writing to file: {e}")
        except KeyError:
            print("No results to write.")
