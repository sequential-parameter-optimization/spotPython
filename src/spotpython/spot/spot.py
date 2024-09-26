from __future__ import annotations

from torch.utils.tensorboard import SummaryWriter
import pickle
import pprint
import os
import copy
import json
from numpy.random import default_rng
from spotpython.design.spacefilling import spacefilling
from spotpython.build.kriging import Kriging
from spotpython.utils.repair import repair_non_numeric
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
from spotpython.utils.init import fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
from spotpython.utils.compare import selectNew
from spotpython.utils.aggregate import aggregate_mean_var, select_distant_points
from spotpython.utils.repair import remove_nan
from spotpython.utils.file import get_experiment_filename
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
            experimental design. If `None`, spotpython's `spacefilling` is used.
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
                        theta_init_zero=True,
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

        self.fun = fun
        if self.fun is None:
            raise Exception("No objective function specified.")
        if not callable(self.fun):
            raise Exception("Objective function is not callable.")

        # 1. fun_control updates:
        # -----------------------
        # Random number generator:
        self.rng = default_rng(self.fun_control["seed"])

        # 2. lower attribute updates:
        # -----------------------
        # if lower is in the fun_control dictionary, use the value of the key "lower" as the lower bound
        if get_control_key_value(control_dict=self.fun_control, key="lower") is not None:
            self.lower = get_control_key_value(control_dict=self.fun_control, key="lower")
        # Number of dimensions is based on lower
        self.k = self.lower.size

        # 3. upper attribute updates:
        # -----------------------
        # if upper is in fun_control dictionary, use the value of the key "upper" as the upper bound
        if get_control_key_value(control_dict=self.fun_control, key="upper") is not None:
            self.upper = get_control_key_value(control_dict=self.fun_control, key="upper")

        # 4. var_type attribute updates:
        # -----------------------
        # self.set_self_attribute("var_type", var_type, self.fun_control)
        self.var_type = self.fun_control["var_type"]
        # Force numeric type as default in every dim:
        # assume all variable types are "num" if "num" is
        # specified less than k times
        if len(self.var_type) < self.k:
            self.var_type = self.var_type * self.k
            logger.warning("All variable types forced to 'num'.")

        # 5. var_name attribute updates:
        # -----------------------
        # self.set_self_attribute("var_name", var_name, self.fun_control)
        self.var_name = self.fun_control["var_name"]
        # use x0, x1, ... as default variable names:
        if self.var_name is None:
            self.var_name = ["x" + str(i) for i in range(len(self.lower))]

        # Reduce dim based on lower == upper logic:
        # modifies lower, upper, var_type, and var_name
        self.to_red_dim()

        # 6. Additional self attributes updates:
        # -----------------------
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

        # Tensorboard:
        self.init_spot_writer()

        # Bounds are internal, because they are functions of self.lower and self.upper
        # and used by the optimizer:
        de_bounds = []
        for j in range(self.lower.size):
            de_bounds.append([self.lower[j], self.upper[j]])
        self.de_bounds = de_bounds

        # Design related information:
        self.design = design
        if design is None:
            self.design = spacefilling(k=self.lower.size, seed=self.fun_control["seed"])
        # self.design_control = {"init_size": 10, "repeats": 1}
        # self.design_control.update(design_control)

        # Optimizer related information:
        self.optimizer = optimizer
        # self.optimizer_control = {"max_iter": 1000, "seed": 125}
        # self.optimizer_control.update(optimizer_control)
        if self.optimizer is None:
            self.optimizer = optimize.differential_evolution

        # Surrogate related information:
        self.surrogate = surrogate
        self.surrogate_control.update({"var_type": self.var_type})
        # Surrogate control updates:
        # The default value for `noise` from the surrogate_control dictionary
        # based on surrogate_control.init() is None. This value is updated
        # to the value of the key "noise" from the fun_control dictionary.
        # If the value is set (i.e., not None), it is not updated.
        if self.surrogate_control["noise"] is None:
            self.surrogate_control.update({"noise": self.fun_control.noise})
        if self.surrogate_control["model_fun_evals"] is None:
            self.surrogate_control.update({"model_fun_evals": self.optimizer_control["max_iter"]})
        # self.optimizer is not None here. If 1) the key "model_optimizer"
        # is still None or 2) a user specified optimizer is provided, update the value of
        # the key "model_optimizer" to the value of self.optimizer.
        if self.surrogate_control["model_optimizer"] is None or optimizer is not None:
            self.surrogate_control.update({"model_optimizer": self.optimizer})

        # if self.surrogate_control["n_theta"] is a string and == isotropic, use 1 theta value:
        if isinstance(self.surrogate_control["n_theta"], str):
            if self.surrogate_control["n_theta"] == "anisotropic":
                surrogate_control.update({"n_theta": self.k})
            else:
                # case "isotropic":
                surrogate_control.update({"n_theta": 1})
        if isinstance(self.surrogate_control["n_theta"], int):
            if self.surrogate_control["n_theta"] > 1:
                surrogate_control.update({"n_theta": self.k})

        # If no surrogate model is specified, use the internal
        # spotpython kriging surrogate:
        if self.surrogate is None:
            # Call kriging with surrogate_control parameters:
            self.surrogate = Kriging(
                name="kriging",
                noise=self.surrogate_control["noise"],
                model_optimizer=self.surrogate_control["model_optimizer"],
                model_fun_evals=self.surrogate_control["model_fun_evals"],
                seed=self.surrogate_control["seed"],
                log_level=self.log_level,
                min_theta=self.surrogate_control["min_theta"],
                max_theta=self.surrogate_control["max_theta"],
                metric_factorial=self.surrogate_control["metric_factorial"],
                n_theta=self.surrogate_control["n_theta"],
                theta_init_zero=self.surrogate_control["theta_init_zero"],
                p_val=self.surrogate_control["p_val"],
                n_p=self.surrogate_control["n_p"],
                optim_p=self.surrogate_control["optim_p"],
                min_Lambda=self.surrogate_control["min_Lambda"],
                max_Lambda=self.surrogate_control["max_Lambda"],
                var_type=self.surrogate_control["var_type"],
                spot_writer=self.spot_writer,
                counter=self.design_control["init_size"] * self.design_control["repeats"] - 1,
            )

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

        logger.setLevel(self.log_level)
        logger.info(f"Starting the logger at level {self.log_level} for module {__name__}:")
        logger.debug("In Spot() init(): fun_control: %s", self.fun_control)
        logger.debug("In Spot() init(): design_control: %s", self.design_control)
        logger.debug("In Spot() init(): optimizer_control: %s", self.optimizer_control)
        logger.debug("In Spot() init(): surrogate_control: %s", self.surrogate_control)
        logger.debug("In Spot() init(): self.get_spot_attributes_as_df(): %s", self.get_spot_attributes_as_df())

    def set_self_attribute(self, attribute, value, dict):
        """
        This function sets the attribute of the 'self' object to the provided value.
        If the key exists in the provided dictionary, it updates the attribute with the value from the dictionary.

        Args:
            self (object): the object whose attribute is to be set
            attribute (str): the attribute to set
            value (Any): the value to set the attribute to
            dict (dict): the dictionary to check for the key
        """
        setattr(self, attribute, value)
        if get_control_key_value(control_dict=dict, key=attribute) is not None:
            setattr(self, attribute, get_control_key_value(control_dict=dict, key=attribute))

    def get_spot_attributes_as_df(self) -> pd.DataFrame:
        """Get all attributes of the spot object as a pandas dataframe.

        Returns:
            (pandas.DataFrame): dataframe with all attributes of the spot object.

        Examples:
            >>> import numpy as np
                from math import inf
                from spotpython.fun.objectivefunctions import analytical
                from spotpython.spot import spot
                                from spotpython.utils.init import (
                    fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
                    )
                # number of initial points:
                ni = 7
                # number of points
                n = 10
                fun = analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1]),
                    upper = np.array([1])
                    fun_evals=n)
                design_control=design_control_init(init_size=ni)
                spot_1 = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control,)
                spot_1.run()
                df = spot_1.get_spot_attributes_as_df()
                df
                    Attribute Name                                    Attribute Value
                0                   X  [[-0.3378148180708981], [0.698908280342222], [...
                1           all_lower                                               [-1]
                2           all_upper                                                [1]
                3        all_var_name                                               [x0]
                4        all_var_type                                              [num]
                5             counter                                                 10
                6           de_bounds                                          [[-1, 1]]
                7              design  <spotpython.design.spacefilling.spacefilling o...
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
                26              noise                                              False
                27         ocba_delta                                                  0
                28  optimizer_control                    {'max_iter': 1000, 'seed': 125}
                29            red_dim                                              False
                30                rng                                   Generator(PCG64)
                31               seed                                                123
                32        show_models                                              False
                33      show_progress                                               True
                34        spot_writer                                               None
                35          surrogate  <spotpython.build.kriging.Kriging object at 0x...
                36  surrogate_control  {'noise': False, 'model_optimizer': <function ...
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
        Reduce dimension if lower == upper.
        This is done by removing the corresponding entries from
        lower, upper, var_type, and var_name.
        k is modified accordingly.

        Args:
            self (object): Spot object

        Returns:
            (NoneType): None

        Attributes:
            self.lower (numpy.ndarray): lower bound
            self.upper (numpy.ndarray): upper bound
            self.var_type (List[str]): list of variable types

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import analytical
                from spotpython.spot import spot
                                from spotpython.utils.init import (
                    fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
                    )
                # number of initial points:
                ni = 3
                # number of points
                n = 10
                fun = analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1, -1]),
                    upper = np.array([1, 1]),
                    fun_evals = n)
                design_control=design_control_init(init_size=ni)
                spot_1 = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control,)
                spot_1.run()
                assert spot_1.lower.size == 2
                assert spot_1.upper.size == 2
                assert len(spot_1.var_type) == 2
                assert spot_1.red_dim == False
                spot_1.lower = np.array([-1, -1])
                spot_1.upper = np.array([-1, -1])
                spot_1.to_red_dim()
                assert spot_1.lower.size == 0
                assert spot_1.upper.size == 0
                assert len(spot_1.var_type) == 0
                assert spot_1.red_dim == True

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
        # Modify k (dim):
        self.k = self.lower.size
        # Modify var_type:
        if self.var_type is not None:
            self.all_var_type = self.var_type
            self.var_type = [x for x, y in zip(self.all_var_type, self.ident) if not y]
        # Modify var_name:
        if self.var_name is not None:
            self.all_var_name = self.var_name
            self.var_name = [x for x, y in zip(self.all_var_name, self.ident) if not y]

    def to_all_dim(self, X0) -> np.array:
        n = X0.shape[0]
        k = len(self.ident)
        X = np.zeros((n, k))
        j = 0
        for i in range(k):
            if self.ident[i]:
                X[:, i] = self.all_lower[i]
                j += 1
            else:
                X[:, i] = X0[:, i - j]
        return X

    def to_all_dim_if_needed(self, X) -> np.array:
        if self.red_dim:
            return self.to_all_dim(X)
        else:
            return X

    def get_new_X0(self) -> np.array:
        """
        Get new design points.
        Calls `suggest_new_X()` and repairs the new design points, e.g.,
        by `repair_non_numeric()` and `selectNew()`.

        Args:
            self (object): Spot object

        Returns:
            (numpy.ndarray): new design points

        Notes:
            * self.design (object): an experimental design is used to generate new design points
            if no new design points are found, a new experimental design is generated.

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import analytical
                               from spotpython.utils.init import (
                    fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
                    )
                from spotpython.spot import spot
                from spotpython.utils.init import fun_control_init
                # number of initial points:
                ni = 3
                X_start = np.array([[0, 1], [1, 0], [1, 1], [1, 1]])
                fun = analytical().fun_sphere
                fun_control = fun_control_init(
                            n_points=10,
                            ocba_delta=0,
                            lower = np.array([-1, -1]),
                            upper = np.array([1, 1])
                )
                design_control=design_control_init(init_size=ni)
                S = spot.Spot(fun=fun,
                                fun_control=fun_control
                                design_control=design_control,
                )
                S.initialize_design(X_start=X_start)
                S.update_stats()
                S.fit_surrogate()
                X_ocba = None
                X0 = S.get_new_X0()
                assert X0.shape[0] == S.n_points
                assert X0.shape[1] == S.lower.size
                # assert new points are in the interval [lower, upper]
                assert np.all(X0 >= S.lower)
                assert np.all(X0 <= S.upper)
                # print using 20 digits precision
                np.set_printoptions(precision=20)
                print(f"X0: {X0}")

        """
        # Try to generate self.fun_repeats new X0 points:
        X0 = self.suggest_new_X()
        X0 = repair_non_numeric(X0, self.var_type)
        # (S-16) Duplicate Handling:
        # Condition: select only X= that have min distance
        # to existing solutions
        X0, X0_ind = selectNew(A=X0, X=self.X, tolerance=self.tolerance_x)
        if X0.shape[0] > 0:
            # 1. There are X0 that fullfil the condition.
            # Note: The number of new X0 can be smaller than self.n_points!
            logger.debug("XO values are new: %s %s", X0_ind, X0)
            return repeat(X0, self.fun_repeats, axis=0)
            return X0
        # 2. No X0 found. Then generate self.n_points new solutions:
        else:
            self.design = spacefilling(k=self.k, seed=self.fun_control["seed"] + self.counter)
            X0 = self.generate_design(
                size=self.n_points, repeats=self.design_control["repeats"], lower=self.lower, upper=self.upper
            )
            X0 = repair_non_numeric(X0, self.var_type)
            logger.warning("No new XO found on surrogate. Generate new solution %s", X0)
            return X0

    def de_serialize_dicts(self) -> tuple:
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

    def write_db_dict(self) -> None:
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
        ) = self.de_serialize_dicts()
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

    def run(self, X_start=None) -> Spot:
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
            self (object): Spot object
            X_start (numpy.ndarray, optional): initial design. Defaults to None.

        Returns:
            (object): Spot object

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import analytical
                from spotpython.spot import spot
                from spotpython.utils.init import (
                    fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
                    )
                # number of initial points:
                ni = 7
                # start point X_0
                X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                fun = analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1, -1]),
                    upper = np.array([1, 1]))
                design_control=design_control_init(init_size=ni)
                S = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control,)
                S.run(X_start=X_start)
                print(f"S.X: {S.X}")
                print(f"S.y: {S.y}")
                Seed set to 123
                Seed set to 123
                spotpython tuning: 0.0 [########--] 80.00%
                spotpython tuning: 0.0 [#########-] 86.67%
                spotpython tuning: 0.0 [#########-] 93.33%
                spotpython tuning: 0.0 [##########] 100.00% Done...

                S.X: [[ 0.00000000e+00  0.00000000e+00]
                [ 0.00000000e+00  1.00000000e+00]
                [ 1.00000000e+00  0.00000000e+00]
                [ 1.00000000e+00  1.00000000e+00]
                [-9.09243389e-01 -1.58234577e-01]
                [-2.05817107e-01 -4.81249089e-01]
                [ 9.49741171e-01 -9.46312716e-01]
                [-1.20955714e-01  6.38358863e-02]
                [-6.62787018e-01  1.74316373e-01]
                [ 2.82008441e-01  9.30010114e-01]
                [ 4.78788115e-01  6.53210582e-01]
                [ 2.64764215e-04  4.00803185e-03]
                [-1.66363820e-05  4.65001027e-03]
                [-2.60995680e-04  5.46114194e-03]
                [ 3.74504308e-03  1.86731890e-02]]
                S.y: [0.00000000e+00 1.00000000e+00 1.00000000e+00 2.00000000e+00
                8.51761723e-01 2.73961367e-01 1.79751605e+00 1.87053051e-02
                4.69672829e-01 9.44447573e-01 6.55922124e-01 1.61344194e-05
                2.16228723e-05 2.98921900e-05 3.62713334e-04]

        """
        self.initialize_design(X_start)
        # New: self.update_stats() moved here:
        # changed in 0.5.9:
        self.update_stats()
        # (S-4): Imputation:
        # Not implemented yet.
        # (S-11) Surrogate Fit:
        self.fit_surrogate()
        # (S-5) Calling the spotLoop Function
        # and
        # (S-9) Termination Criteria, Conditions:
        timeout_start = time.time()
        while self.should_continue(timeout_start):
            self.update_design()
            # (S-10): Subset Selection for the Surrogate:
            # Not implemented yet.
            # Update stats
            self.update_stats()
            # Update writer:
            self.update_writer()
            # (S-11) Surrogate Fit:
            self.fit_surrogate()
            # progress bar:
            self.show_progress_if_needed(timeout_start)
        if hasattr(self, "spot_writer") and self.spot_writer is not None:
            self.spot_writer.flush()
            self.spot_writer.close()
        if self.fun_control["db_dict_name"] is not None:
            self.write_db_dict()
        if self.fun_control["save_experiment"]:
            self.save_experiment()
        return self

    def initialize_design(self, X_start=None) -> None:
        """
        Initialize design. Generate and evaluate initial design.
        If `X_start` is not `None`, append it to the initial design.
        Therefore, the design size is `init_size` + `X_start.shape[0]`.

        Args:
            self (object): Spot object
            X_start (numpy.ndarray, optional): initial design. Defaults to None.

        Returns:
            (NoneType): None

        Attributes:
            self.X (numpy.ndarray): initial design
            self.y (numpy.ndarray): initial design values

        Note:
            * If `X_start` is has the wrong shape, it is ignored.

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import analytical
                from spotpython.spot import spot
                from spotpython.utils.init import (
                    fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
                    )
                # number of initial points:
                ni = 7
                # start point X_0
                X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                fun = analytical().fun_sphere
                fun_control = fun_control_init(
                    lower = np.array([-1, -1]),
                    upper = np.array([1, 1]))
                design_control=design_control_init(init_size=ni)
                S = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control,)
                S.initialize_design(X_start=X_start)
                print(f"S.X: {S.X}")
                print(f"S.y: {S.y}")
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
                    X0 = append(X_start, X0, axis=0)
                else:
                    X0 = X_start
            except ValueError:
                logger.warning("X_start has wrong shape. Ignoring it.")
        if X0.shape[0] == 0:
            raise Exception("X0 has zero rows. Check design_control['init_size'] or X_start.")
        X0 = repair_non_numeric(X0, self.var_type)
        self.X = X0
        # (S-3): Eval initial design:
        X_all = self.to_all_dim_if_needed(X0)
        logger.debug("In Spot() initialize_design(), before calling self.fun: X_all: %s", X_all)
        logger.debug("In Spot() initialize_design(), before calling self.fun: fun_control: %s", self.fun_control)
        self.y = self.fun(X=X_all, fun_control=self.fun_control)
        logger.debug("In Spot() initialize_design(), after calling self.fun: self.y: %s", self.y)
        # TODO: Error if only nan values are returned
        logger.debug("New y value: %s", self.y)
        #
        self.counter = self.y.size
        if hasattr(self, "spot_writer") and self.spot_writer is not None:
            # range goes to init_size -1 because the last value is added by update_stats(),
            # which always adds the last value.
            # Changed in 0.5.9:
            for j in range(len(self.y)):
                X_j = self.X[j].copy()
                y_j = self.y[j].copy()
                config = {self.var_name[i]: X_j[i] for i in range(self.k)}
                # see: https://github.com/pytorch/pytorch/issues/32651
                # self.spot_writer.add_hparams(config, {"spot_y": y_j}, run_name=self.spot_tensorboard_path)
                self.spot_writer.add_hparams(config, {"spot_y": y_j})
                self.spot_writer.flush()
        #
        self.X, self.y = remove_nan(self.X, self.y, stop_on_zero_return=True)
        logger.debug("In Spot() initialize_design(), final X val, after remove nan: self.X: %s", self.X)
        logger.debug("In Spot() initialize_design(), final y val, after remove nan: self.y: %s", self.y)

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
                from spotpython.fun.objectivefunctions import analytical
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
        X0 = repair_non_numeric(X0, self.var_type)
        X_all = self.to_all_dim_if_needed(X0)
        logger.debug("In Spot() generate_random_point(), before calling self.fun: X_all: %s", X_all)
        logger.debug("In Spot() generate_random_point(), before calling self.fun: fun_control: %s", self.fun_control)
        y0 = self.fun(X=X_all, fun_control=self.fun_control)
        X0, y0 = remove_nan(X0, y0, stop_on_zero_return=False)
        return X0, y0

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
                from spotpython.fun.objectivefunctions import analytical
                from spotpython.utils.init import (
                    fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
                    )
                from spotpython.spot import spot
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
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import analytical
                from spotpython.spot import spot
                from spotpython.utils.init import fun_control_init
                # number of initial points:
                ni = 3
                X_start = np.array([[0, 1], [1, 0], [1, 1], [1, 1]])
                fun = analytical().fun_sphere
                fun_control = fun_control_init(
                        sigma=0.02,
                        lower = np.array([-1, -1]),
                        upper = np.array([1, 1]),
                        noise=True,
                        ocba_delta=1,
                    )
                design_control=design_control_init(init_size=ni, repeats=2)

                S = spot.Spot(fun=fun,
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
                assert X_shape_before[0] + S.n_points * S.fun_repeats + S.ocba_delta == S.X.shape[0]
                assert X_shape_before[1] == S.X.shape[1]
                assert y_size_before + S.n_points * S.fun_repeats + S.ocba_delta == S.y.size

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
        y0 = self.fun(X=X_all, fun_control=self.fun_control)
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

    def fit_surrogate(self) -> None:
        """
        Fit surrogate model. The surrogate model
        is fitted to the data stored in `self.X` and `self.y`.
        It uses the generic `fit()` method of the
        surrogate model `surrogate`. The default surrogate model is
        an instance from spotpython's `Kriging` class.
        Args:
            self (object): Spot object

        Returns:
            (NoneType): None

        Attributes:
            self.surrogate (object): surrogate model

        Note:
            * As shown in https://sequential-parameter-optimization.github.io/Hyperparameter-Tuning-Cookbook/
            other surrogate models can be used as well.

        Examples:
                >>> import numpy as np
                    from spotpython.fun.objectivefunctions import analytical
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

        """
        logger.debug("In fit_surrogate(): self.X: %s", self.X)
        logger.debug("In fit_surrogate(): self.y: %s", self.y)
        logger.debug("In fit_surrogate(): self.X.shape: %s", self.X.shape)
        logger.debug("In fit_surrogate(): self.y.shape: %s", self.y.shape)
        X_points = self.X.shape[0]
        y_points = self.y.shape[0]
        if X_points == y_points:
            if X_points > self.max_surrogate_points:
                X_S, y_S = select_distant_points(X=self.X, y=self.y, k=self.max_surrogate_points)
            else:
                X_S = self.X
                y_S = self.y
            self.surrogate.fit(X_S, y_S)
        else:
            logger.warning("X and y have different sizes. Surrogate not fitted.")
        if self.show_models:
            self.plot_model()

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
            progress_bar(
                progress=(time.time() - timeout_start) / (self.max_time * 60), y=self.min_y, filename=self.progress_file
            )

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
                from spotpython.fun.objectivefunctions import analytical
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

    def update_stats(self) -> None:
        """
        Update the following stats: 1. `min_y` 2. `min_X` 3. `counter`
        If `noise` is `True`, additionally the following stats are computed: 1. `mean_X`
        2. `mean_y` 3. `min_mean_y` 4. `min_mean_X`.

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
            # variance of the best mean y value so far:
            self.min_var_y = self.var_y[argmin(self.mean_y)]
            # best mean y value so far:
            self.min_mean_y = self.mean_y[argmin(self.mean_y)]

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
                self.spot_writer.add_scalars(
                    "spot_y", {f"y_last_n{i}": y_last_n[i] for i in range(self.fun_repeats)}, self.counter
                )
                # X_min_mean: X value of the best mean y value so far (see above)
                self.spot_writer.add_scalars(
                    "spot_X_noise", {f"X_min_mean{i}": X_min_mean[i] for i in range(self.k)}, self.counter
                )
            # get last value of self.X and convert to dict. take the values from self.var_name as keys:
            X_last = self.X[-1].copy()
            config = {self.var_name[i]: X_last[i] for i in range(self.k)}
            # hyperparameters X and value y of the last configuration:
            # see: https://github.com/pytorch/pytorch/issues/32651
            # self.spot_writer.add_hparams(config, {"spot_y": y_last}, run_name=self.spot_tensorboard_path)
            self.spot_writer.add_hparams(config, {"spot_y": y_last})
            self.spot_writer.flush()

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
                from spotpython.spot import spot
                from spotpython.fun.objectivefunctions import analytical
                from spotpython.utils.init import (
                    fun_control_init, optimizer_control_init, surrogate_control_init, design_control_init
                    )
                nn = 3
                fun_sphere = analytical().fun_sphere
                fun_control = fun_control_init(
                        lower = np.array([-1, -1]),
                        upper = np.array([1, 1]),
                        n_points=nn,
                        )
                spot_1 = spot.Spot(
                    fun=fun_sphere,
                    fun_control=fun_control,
                    )
                # (S-2) Initial Design:
                spot_1.X = spot_1.design.scipy_lhd(
                    spot_1.design_control["init_size"], lower=spot_1.lower, upper=spot_1.upper
                )
                print(f"spot_1.X: {spot_1.X}")
                # (S-3): Eval initial design:
                spot_1.y = spot_1.fun(spot_1.X)
                print(f"spot_1.y: {spot_1.y}")
                spot_1.fit_surrogate()
                X0 = spot_1.suggest_new_X()
                print(f"X0: {X0}")
                assert X0.size == spot_1.n_points * spot_1.k
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
            "basinhopping": lambda: self.optimizer(func=self.infill, x0=self.min_X),
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
        if isinstance(self.surrogate, Kriging):
            return self.surrogate.predict(X, return_val=self.infill_criterion)
        else:
            return self.surrogate.predict(X)

    def plot_progress(
        self, show=True, log_x=False, log_y=False, filename="plot.png", style=["ko", "k", "ro-"], dpi=300, tkagg=False
    ) -> None:
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
                from spotpython.fun.objectivefunctions import analytical
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
                from spotpython.fun.objectivefunctions import analytical
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
            y_test = self.fun(X=X_test.reshape(-1, 1), fun_control=self.fun_control)
            if isinstance(self.surrogate, Kriging):
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
                plt.title(
                    "fun_evals: "
                    + str(self.counter)
                    + ". min_y (noise): "
                    + str(np.round(self.min_y, 6))
                    + " min_mean_y: "
                    + str(np.round(self.min_mean_y, 6))
                )
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

    def print_results_old(self, print_screen=True, dict=None) -> list[str]:
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
        if self.noise:
            res = self.to_all_dim(self.min_mean_X.reshape(1, -1))
            if print_screen:
                print(f"min mean y: {self.min_mean_y}")
            for i in range(res.shape[1]):
                var_name = "x" + str(i) if self.all_var_name is None else self.all_var_name[i]
                if print_screen:
                    print(var_name + ":", res[0][i])
                output.append([var_name, res[0][i]])
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
                surrogate_control = surrogate_control_init(noise=True,
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
                    val = get_ith_hyperparameter_name_from_fun_control(
                        fun_control=fun_control, key=var_name, i=int(res[0][i])
                    )
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
                    from spotpython.fun.objectivefunctions import analytical
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
        ax_3d.plot_surface(
            X_combined, Y_combined, Z_combined, rstride=3, cstride=3, alpha=0.9, cmap="jet", vmin=min_z, vmax=max_z
        )

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
                from spotpython.fun.objectivefunctions import analytical
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
        print(f"impo: {impo}")
        indices = sort_by_kth_and_return_indices(array=impo, k=1)
        print(f"indices: {indices}")
        # take the first max_imp values from the indices array
        if max_imp is not None:
            indices = indices[:max_imp]
        print(f"indices after max_imp selection: {indices}")
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
                from spotpython.fun.objectivefunctions import analytical
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
                dimensions=list(
                    [
                        dict(range=[min(df.iloc[:, i]), max(df.iloc[:, i])], label=df.columns[i], values=df.iloc[:, i])
                        for i in range(len(df.columns) - 1)
                    ]
                ),
            )
        )
        if show:
            fig.show()
        return fig

    def save_experiment(self, filename=None, overwrite=True) -> None:
        """
        Save the experiment to a file.

        Args:
            filename (str):
                The filename of the experiment file.
            overwrite (bool):
                If `True`, the file will be overwritten if it already exists. Default is `True`.

        Returns:
            None
        """
        # Remove or close any unpickleable objects, e.g., the spot_writer
        self.close_and_del_spot_writer()

        # Remove the logger handler before pickling
        self.remove_logger_handlers()

        # Create deep copies of control dictionaries
        fun_control = copy.deepcopy(self.fun_control)
        optimizer_control = copy.deepcopy(self.optimizer_control)
        surrogate_control = copy.deepcopy(self.surrogate_control)
        design_control = copy.deepcopy(self.design_control)

        # Deep copy the spot object itself (except unpickleable components)
        try:
            spot_tuner = copy.deepcopy(self)
        except Exception as e:
            print("Warning: Could not copy spot_tuner object!")
            print(f"Error: {e}")
            spot_tuner = self

        # Prepare the experiment dictionary
        experiment = {
            "design_control": design_control,
            "fun_control": fun_control,
            "optimizer_control": optimizer_control,
            "spot_tuner": spot_tuner,
            "surrogate_control": surrogate_control,
        }

        # Determine the filename based on PREFIX if not provided
        PREFIX = fun_control.get("PREFIX")
        if filename is None and PREFIX is not None:
            filename = get_experiment_filename(PREFIX)

        # Check if the file already exists
        if filename is not None and os.path.exists(filename) and not overwrite:
            print(f"Error: File {filename} already exists. Use overwrite=True to overwrite the file.")
            return

        # Serialize the experiment dictionary to the pickle file
        if filename is not None:
            with open(filename, "wb") as handle:
                try:
                    pickle.dump(experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f"Error: {e}")
                    raise e
            print(f"Experiment saved to {filename}")

    def remove_logger_handlers(self) -> None:
        """
        Remove handlers from the logger to avoid pickling issues.
        """
        logger = logging.getLogger(__name__)
        for handler in logger.handlers[:]:  # Copy the list to avoid modification during iteration
            logger.removeHandler(handler)

    def reattach_logger_handlers(self) -> None:
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

    def init_spot_writer(self) -> None:
        """
        Initialize the spot_writer for the current experiment if in fun_control
        the tensorboard_log is set to True
        and the spot_tensorboard_path is not None.
        Otherwise, the spot_writer is set to None.
        """
        if self.fun_control["tensorboard_log"] and self.fun_control["spot_tensorboard_path"] is not None:
            self.spot_writer = SummaryWriter(log_dir=self.fun_control["spot_tensorboard_path"])
        else:
            self.spot_writer = None

    def close_and_del_spot_writer(self) -> None:
        """
        Delete the spot_writer attribute from the object
        if it exists and close the writer.
        """
        if hasattr(self, "spot_writer") and self.spot_writer is not None:
            self.spot_writer.flush()
            self.spot_writer.close()
            del self.spot_writer
