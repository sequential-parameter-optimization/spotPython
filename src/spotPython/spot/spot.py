from numpy.random import default_rng
from ..design.spacefilling import spacefilling
from ..build.kriging import Kriging
from ..utils.repair import repair_non_numeric
import numpy as np
import pandas as pd
import pylab
from scipy import optimize
from scipy.optimize import differential_evolution
from math import inf
from math import isfinite
import matplotlib.pyplot as plt
from numpy import argmin
from numpy import repeat
from numpy import sqrt
from numpy import spacing
from numpy import meshgrid
from numpy import ravel
from numpy import array
from numpy import append
from spotPython.utils.compare import selectNew
from spotPython.utils.aggregate import aggregate_mean_var
from spotPython.utils.repair import remove_nan
from spotPython.budget.ocba import get_ocba_X
import logging
import time
from spotPython.utils.progress import progress_bar
from spotPython.utils.convert import find_indices
import plotly.graph_objects as go


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

    * Getting and setting parameters. This is done via the `Spot` initilaization.
    * Running surrogate based hyperparameter optimization. After the class is initialized, hyperparameter tuning
    runs can be performed via the `run` method.
    * Displaying information. The `plot` method can be used for visualizing results. The `print` methods summarizes
    information about the tuning run.

    The `Spot` class is build in a modular manner. It combines the following three components:

        1. Design
        2. Surrogate
        3. Optimizer

    For each of the three components different implementations can be selected and combined. Internal components
    are selected as default. These can be replaced by components from other packages, e.g., scikit-learn or
    scikit-optimize.

    Args:
        fun (object): objective function
        lower (numpy.array): lower bound
        upper (numpy.array): upper bound
        fun_evals (int):
            number of function evaluations
        fun_repeats (int):
            number of repeats (replicates).
        max_time (int):
            maximum time (in minutes)
        noise (bool):
            deterministic or noisy objective function
        tolerance_x (float):
            tolerance for new x solutions. Minimum distance of new solutions,
            generated by `suggest_new_X`, to already existing solutions.
            If zero (which is the default), every new solution is accepted.
        ocba_delta (int): OCBA increment (only used if `noise==True`)
        var_type (list): list of type information, can be either "num" or "factor"
        var_name (list): list of variable names, e.g., ["x1", "x2"]
        infill_criterion (string): Can be `"y"`, `"s"`, `"ei"` (negative expected improvement), or `"all"`.
        n_points (int): number of infill points
        seed (int): initial seed
        log_level (int): log level with the following settings:
            `NOTSET` (`0`),
            `DEBUG` (`10`: Detailed information, typically of interest only when diagnosing problems.),
            `INFO` (`20`: Confirmation that things are working as expected.),
            `WARNING` (`30`: An indication that something unexpected happened, or indicative of some problem in the near
                future (e.g. ‘disk space low’). The software is still working as expected.),
            `ERROR` (`40`: Due to a more serious problem, the software has not been able to perform some function.), and
            `CRITICAL` (`50`: A serious error, indicating that the program itself may be unable to continue running.)
        show_models (bool): Plot model. Currently only 1-dim functions are supported.
        show_progress (bool). Show progress bar.
        design (object): experimental design.
        design_control (dict): experimental design information stored as a dictionary with the following entries:
            "init_size": `10`, "repeats": `1`.
        surrogate (object): surrogate model. If `None`, spotPython's `kriging` is used.
        surrogate_control (dict): surrogate model information stored as a dictionary with the following entries:
            "model_optimizer": `differential_evolution`,
            "model_fun_evals": `None`,
            "min_theta": `-3.`,
            "max_theta": `3.`,
            "n_theta": `1`,
            "n_p": `1`,
            "optim_p": `False`,
            "cod_type": `"norm"`,
            "var_type": `self.var_type`,
            "use_cod_y": `False`.
        optimizer (object): optimizer. If `None`, `scipy.optimize`'s `differential_evolution` is used.
        optimizer_control (dict): information about the optimizer stored as a dictionary with the following entries:
            "max_iter": `1000`.
    Returns:
        object: `Spot` object

    Note:
        Description in the source code refers to [bart21i]:
        Bartz-Beielstein, T., and Zaefferer, M. Hyperparameter tuning approaches.
        In Hyperparameter Tuning for Machine and Deep Learning with R - A Practical Guide,
        E. Bartz, T. Bartz-Beielstein, M. Zaefferer, and O. Mersmann, Eds. Springer, 2022, ch. 4, pp. 67–114.
    """

    def __str__(self):
        return self.__class__.__name__

    def __init__(
        self,
        fun,
        lower,
        upper,
        fun_evals=15,
        fun_repeats=1,
        fun_control={},
        max_time=inf,
        noise=False,
        tolerance_x=0,
        var_type=["num"],
        var_name=None,
        all_var_name=None,
        infill_criterion="y",
        n_points=1,
        ocba_delta=0,
        seed=123,
        log_level=50,
        show_models=False,
        show_progress=False,
        design=None,
        design_control={},
        surrogate=None,
        surrogate_control={},
        optimizer=None,
        optimizer_control={},
    ):
        # small value:
        self.eps = sqrt(spacing(1))
        self.fun = fun
        self.lower = lower
        self.upper = upper
        self.var_type = var_type
        self.var_name = var_name
        self.all_var_name = all_var_name
        # Reduce dim based on lower == upper logic:
        # modifies lower, upper, and var_type
        self.to_red_dim()
        self.k = self.lower.size
        self.fun_evals = fun_evals
        self.fun_repeats = fun_repeats
        self.max_time = max_time
        self.noise = noise
        self.tolerance_x = tolerance_x
        self.ocba_delta = ocba_delta
        self.log_level = log_level
        self.show_models = show_models
        self.show_progress = show_progress
        # Random number generator:
        self.seed = seed
        self.rng = default_rng(self.seed)
        # Force numeric type as default in every dim:
        # assume all variable types are "num" if "num" is
        # specified once:
        if len(self.var_type) < self.k:
            self.var_type = self.var_type * self.k
            logger.warning("Warning: All variable types forced to 'num'.")
        self.infill_criterion = infill_criterion
        # Bounds
        de_bounds = []
        for j in range(self.k):
            de_bounds.append([self.lower[j], self.upper[j]])
        self.de_bounds = de_bounds
        # Infill points:
        self.n_points = n_points
        # Objective function related information:
        self.fun_control = {"sigma": 0, "seed": None}
        self.fun_control.update(fun_control)
        # Design related information:
        self.design = design
        if design is None:
            self.design = spacefilling(k=self.k, seed=self.seed)
        self.design_control = {"init_size": 10, "repeats": 1}
        self.design_control.update(design_control)
        # Surrogate related information:
        self.surrogate = surrogate
        self.surrogate_control = {
            "noise": self.noise,
            "model_optimizer": differential_evolution,
            "model_fun_evals": None,
            "min_theta": -3.0,
            "max_theta": 3.0,
            "n_theta": 1,
            "n_p": 1,
            "optim_p": False,
            "cod_type": "norm",
            "var_type": self.var_type,
            "seed": 124,
            "use_cod_y": False,
        }
        self.surrogate_control.update(surrogate_control)
        # If no surrogate model is specified, use the internal
        # spotPython kriging surrogate:
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
                n_theta=self.surrogate_control["n_theta"],
                n_p=self.surrogate_control["n_p"],
                optim_p=self.surrogate_control["optim_p"],
                cod_type=self.surrogate_control["cod_type"],
                var_type=self.surrogate_control["var_type"],
                use_cod_y=self.surrogate_control["use_cod_y"],
            )
        # Optimizer related information:
        self.optimizer = optimizer
        self.optimizer_control = {"max_iter": 1000, "seed": 125}
        self.optimizer_control.update(optimizer_control)
        if self.optimizer is None:
            self.optimizer = optimize.differential_evolution
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

    def to_red_dim(self):
        self.all_lower = self.lower
        self.all_upper = self.upper
        self.ident = (self.upper - self.lower) == 0
        self.lower = self.lower[~self.ident]
        self.upper = self.upper[~self.ident]
        self.red_dim = self.ident.any()
        self.all_var_type = self.var_type
        self.var_type = [x for x, y in zip(self.all_var_type, self.ident) if not y]
        if self.var_name is not None:
            self.all_var_name = self.var_name
            self.var_name = [x for x, y in zip(self.all_var_name, self.ident) if not y]

    def to_all_dim(self, X0):
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

    def to_all_dim_if_needed(self, X):
        if self.red_dim:
            return self.to_all_dim(X)
        else:
            return X

    def get_X_ocba(self):
        if self.noise and self.ocba_delta > 0:
            return get_ocba_X(self.mean_X, self.mean_y, self.var_y, self.ocba_delta)
        else:
            return None

    def get_new_X0(self):
        X0 = self.suggest_new_X()
        X0 = repair_non_numeric(X0, self.var_type)
        # (S-16) Duplicate Handling:
        # Condition: select only X= that have min distance
        # to existing solutions
        X0, X0_ind = selectNew(A=X0, X=self.X, tolerance=self.tolerance_x)
        logger.debug("XO values are new: %s %s", X0_ind, X0)
        # 1. There are X0 that fullfil the condition.
        # Note: The number of new X0 can be smaller than self.n_points!
        if X0.shape[0] > 0:
            return repeat(X0, self.fun_repeats, axis=0)
        # 2. No X0 found. Then generate self.n_points new solutions:
        else:
            self.design = spacefilling(k=self.k, seed=self.seed + self.counter)
            X0 = self.generate_design(
                size=self.n_points, repeats=self.design_control["repeats"], lower=self.lower, upper=self.upper
            )
            X0 = repair_non_numeric(X0, self.var_type)
            logger.warning("No new XO found on surrogate. Generate new solution %s", X0)
            return X0

    def append_X_ocba(self, X_ocba, X0):
        if self.noise and self.ocba_delta > 0:
            return append(X_ocba, X0, axis=0)
        else:
            return X0

    def run(self, X_start=None):
        self.initialize_design(X_start)
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
            # (S-11) Surrogate Fit:
            self.fit_surrogate()
            # progress bar:
            self.show_progress_if_needed(timeout_start)
        return self

    def initialize_design(self, X_start=None):
        # (S-2) Initial Design:
        X0 = self.generate_design(
            size=self.design_control["init_size"],
            repeats=self.design_control["repeats"],
            lower=self.lower,
            upper=self.upper,
        )
        if X_start is not None:
            try:
                X0 = append(X_start, X0, axis=0)
            except ValueError:
                logger.warning("X_start has wrong shape. Ignoring it.")
        X0 = repair_non_numeric(X0, self.var_type)
        self.X = X0
        # (S-3): Eval initial design:
        X_all = self.to_all_dim_if_needed(X0)
        self.y = self.fun(X=X_all, fun_control=self.fun_control)
        # TODO: Error if only nan values are returned
        logger.debug("New y value: %s", self.y)
        self.X, self.y = remove_nan(self.X, self.y)
        self.update_stats()
        # (S-4): Imputation:
        # Not implemented yet.
        # (S-11) Surrogate Fit:
        self.fit_surrogate()

    def should_continue(self, timeout_start):
        return (self.counter < self.fun_evals) and (time.time() < timeout_start + self.max_time * 60)

    def update_design(self):
        # OCBA (only if noise)
        X_ocba = self.get_X_ocba()
        # (S-15) Compile Surrogate Results:
        X0 = self.get_new_X0()
        # (S-18): Evaluating New Solutions:
        X0 = self.append_X_ocba(X_ocba, X0)
        X_all = self.to_all_dim_if_needed(X0)
        y0 = self.fun(X=X_all, fun_control=self.fun_control)
        X0, y0 = remove_nan(X0, y0)
        # Append New Solutions:
        self.X = np.append(self.X, X0, axis=0)
        self.y = np.append(self.y, y0)

    def fit_surrogate(self):
        self.surrogate.fit(self.X, self.y)
        if self.show_models:
            self.plot_model()

    def show_progress_if_needed(self, timeout_start):
        if not self.show_progress:
            return
        if isfinite(self.fun_evals):
            progress_bar(progress=self.counter / self.fun_evals)
        else:
            progress_bar(progress=(time.time() - timeout_start) / (self.max_time * 60))

    def generate_design(self, size, repeats, lower, upper):
        return self.design.scipy_lhd(n=size, repeats=repeats, lower=lower, upper=upper)

    def update_stats(self):
        """
        Update the following stats: 1. `min_y` 2. `min_X` 3. `counter`
        If `noise` is `True`, additionally the following stats are computed: 1. `mean_X`
        2. `mean_y` 3. `min_mean_y` 4. `min_mean_X`.

        """
        self.min_y = min(self.y)
        self.min_X = self.X[argmin(self.y)]
        self.counter = self.y.size
        # Update aggregated x and y values (if noise):
        if self.noise:
            Z = aggregate_mean_var(X=self.X, y=self.y)
            self.mean_X = Z[0]
            self.mean_y = Z[1]
            self.var_y = Z[2]
            self.min_mean_y = min(self.mean_y)
            self.min_mean_X = self.mean_X[argmin(self.mean_y)]

    def suggest_new_X_old(self):
        """
        Compute `n_points` new infill points in natural units.
        The optimizer searches in the ranges from `lower_j` to `upper_j`.
        The method `infill()` is used as the objective function.

        Returns:
            (numpy.ndarray): `n_points` infill points in natural units, each of dim k

        Note:
            This is step (S-14a) in [bart21i].
        """
        # (S-14a) Optimization on the surrogate:
        new_X = np.zeros([self.n_points, self.k], dtype=float)

        optimizer_name = self.optimizer.__name__
        for i in range(self.n_points):
            if optimizer_name == "dual_annealing":
                result = self.optimizer(func=self.infill, bounds=self.de_bounds)
            elif optimizer_name == "differential_evolution":
                result = self.optimizer(
                    func=self.infill,
                    bounds=self.de_bounds,
                    maxiter=self.optimizer_control["max_iter"],
                    seed=self.optimizer_control["seed"],
                    # popsize=10,
                    # updating="deferred"
                )
            elif optimizer_name == "direct":
                result = self.optimizer(func=self.infill, bounds=self.de_bounds, eps=1e-2)
            elif optimizer_name == "shgo":
                result = self.optimizer(func=self.infill, bounds=self.de_bounds)
            elif optimizer_name == "basinhopping":
                result = self.optimizer(func=self.infill, x0=self.min_X)
            else:
                result = self.optimizer(func=self.infill, bounds=self.de_bounds)
            new_X[i][:] = result.x
        return new_X

    def suggest_new_X(self):
        """
        Compute `n_points` new infill points in natural units.
        The optimizer searches in the ranges from `lower_j` to `upper_j`.
        The method `infill()` is used as the objective function.

        Returns:
            (numpy.ndarray): `n_points` infill points in natural units, each of dim k

        Note:
            This is step (S-14a) in [bart21i].
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
            result = optimizers.get(optimizer_name, optimizers["default"])()
            new_X[i][:] = result.x
        return new_X

    def infill(self, x):
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
        x_reshaped = x.reshape(1, -1)
        if isinstance(self.surrogate, Kriging):
            return self.surrogate.predict(x_reshaped, return_val=self.infill_criterion)
        else:
            return self.surrogate.predict(x_reshaped)

    def plot_progress(
        self, show=True, log_x=False, log_y=False, filename="plot.png", style=["ko", "k", "ro-"], dpi=300
    ) -> None:
        """Plot the progress of the hyperparameter tuning (optimization).
        Args:
            show (bool): Show the plot.
            log_x (bool): Use logarithmic scale for x-axis.
            log_y (bool): Use logarithmic scale for y-axis.
            filename (str): Filename to save the plot.
            style (list): Style of the plot. Default: ['k', 'ro-'], i.e., the initial points are plotted as a black line
            and the subsequent points as red dots connected by a line.
        Returns:
            None
        """
        fig = pylab.figure(figsize=(9, 6))
        s_y = pd.Series(self.y)
        s_c = s_y.cummin()
        n_init = self.design_control["init_size"] * self.design_control["repeats"]
        ax = fig.add_subplot(211)
        ax.plot(
            range(1, n_init + 1),
            s_y[:n_init],
            style[0],
            range(1, n_init + 1),
            [s_c[:n_init].min()] * n_init,
            style[1],
            range(n_init, len(s_c)),
            s_c[n_init:],
            style[2],
        )
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        if filename is not None:
            pylab.savefig(filename, dpi=dpi, bbox_inches="tight")
        if show:
            pylab.show()

    def plot_model(self, y_min=None, y_max=None):
        """
        Plot the model fit for 1-dim objective functions.

        Args:
            y_min (float, optional): y range, lower bound.
            y_max (float, optional): y range, upper bound.
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
                y_min = min(min(self.y), min(y_test))
            if y_max is None:
                y_max = max(max(self.y), max(y_test))
            plt.ylim((y_min, y_max))
            plt.legend(loc="best")
            # plt.title(self.surrogate.__class__.__name__ + ". " + str(self.counter) + ": " + str(self.min_y))
            if self.noise:
                plt.title(
                    str(self.counter)
                    + ". y (noise): "
                    + str(np.round(self.min_y, 6))
                    + " y mean: "
                    + str(np.round(self.min_mean_y, 6))
                )
            else:
                plt.title(str(self.counter) + ". y: " + str(np.round(self.min_y, 6)))
            plt.show()

    def print_results(self, print_screen=True) -> list[str]:
        """Print results from the run:
            1. min y
            2. min X
            If `noise == True`, additionally the following values are printed:
            3. min mean y
            4. min mean X
        Args:
            print_screen (bool, optional): print results to screen
        Returns:
            output (list): list of results
        """
        output = []
        if print_screen:
            print(f"min y: {self.min_y}")
        res = self.to_all_dim(self.min_X.reshape(1, -1))
        for i in range(res.shape[1]):
            var_name = "x" + str(i) if self.all_var_name is None else self.all_var_name[i]
            if print_screen:
                print(var_name + ":", res[0][i])
            output.append([var_name, res[0][i]])
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

    def chg(self, x, y, z0, i, j):
        """
        Change the values of elements at indices `i` and `j` in the array `z0` to `x` and `y`, respectively.

        Args:
            x (int or float): The new value for the element at index `i`.
            y (int or float): The new value for the element at index `j`.
            z0 (list or numpy.ndarray): The array to be modified.
            i (int): The index of the element to be changed to `x`.
            j (int): The index of the element to be changed to `y`.

        Returns:
            list or numpy.ndarray: The modified array.

        Example:
            >>> z0 = [1, 2, 3]
            >>> chg(4, 5, z0, 0, 2)
            [4, 2, 5]
        """
        z0[i] = x
        z0[j] = y
        return z0

    def plot_contour(
        self, i=0, j=1, min_z=None, max_z=None, show=True, filename=None, n_grid=25, contour_levels=10, dpi=200
    ) -> None:
        """Plot the contour of any dimension.
        Args:
            i (int): the first dimension
            j (int): the second dimension
            min_z (float): the minimum value of z
            max_z (float): the maximum value of z
            show (bool): show the plot
            filename (str): save the plot to a file
            n_grid (int): number of grid points
            contour_levels (int): number of contour levels
        Returns:
            None
        """
        fig = pylab.figure(figsize=(9, 6))
        # lower and upper
        x = np.linspace(self.lower[i], self.upper[i], num=n_grid)
        y = np.linspace(self.lower[j], self.upper[j], num=n_grid)
        X, Y = meshgrid(x, y)
        # Predict based on the optimized results
        z0 = np.mean(np.array([self.lower, self.upper]), axis=0)
        zz = array([self.surrogate.predict(array([self.chg(x, y, z0, i, j)])) for x, y in zip(ravel(X), ravel(Y))])
        zs = zz[:, 0]
        Z = zs.reshape(X.shape)
        if min_z is None:
            min_z = np.min(Z)
        if max_z is None:
            max_z = np.max(Z)
        ax = fig.add_subplot(221)
        # plot predicted values:
        plt.contourf(X, Y, Z, contour_levels, zorder=1, cmap="jet", vmin=min_z, vmax=max_z)
        if self.var_name is None:
            plt.xlabel("x" + str(i))
            plt.ylabel("x" + str(j))
        else:
            plt.xlabel("x" + str(i) + ": " + self.var_name[i])
            plt.ylabel("x" + str(j) + ": " + self.var_name[j])
        plt.title("Surrogate")
        pylab.colorbar()
        ax = fig.add_subplot(222, projection="3d")
        ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.9, cmap="jet", vmin=min_z, vmax=max_z)
        if self.var_name is None:
            plt.xlabel("x" + str(i))
            plt.ylabel("x" + str(j))
        else:
            plt.xlabel("x" + str(i) + ": " + self.var_name[i])
            plt.ylabel("x" + str(j) + ": " + self.var_name[j])
        if filename:
            pylab.savefig(filename, bbox_inches="tight", dpi=dpi, pad_inches=0),
        if show:
            pylab.show()

    def get_importance(self) -> list:
        """Get importance of each variable and return the results as a list.
        Returns:
            output (list): list of results
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

    def print_importance(self, threshold=0.1, print_screen=True) -> list:
        """Print importance of each variable and return the results as a list.
        Args:
            threshold (float): threshold for printing
            print_screen (boolean): if `True`, values are also printed on the screen. Default is `True`.
        Returns:
            output (list): list of results
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

    def plot_importance(self, threshold=0.1, filename=None, dpi=300) -> None:
        """Plot the importance of each variable.
        Args:
            threshold (float):  The threshold of the importance.
            filename (str): The filename of the plot.
        Returns:
            None
        """
        if self.surrogate.n_theta > 1:
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
            plt.show()

    def parallel_plot(self):
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
        fig.show()
