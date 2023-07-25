import copy
from math import erf
import matplotlib.pyplot as plt
from numpy import min, std, var, mean
from numpy import sqrt
from numpy import exp
from numpy import array
from numpy import log
from numpy import power
from numpy import abs
from numpy import sum
from numpy import diag
from numpy import pi
from numpy import ones, zeros
from numpy import spacing, empty_like
from numpy import float64
from numpy import append, ndarray, isinf, linspace, meshgrid, ravel, diag_indices_from, empty
from numpy.linalg import cholesky, solve, LinAlgError, cond
from scipy.optimize import differential_evolution
from scipy.linalg import cholesky as scipy_cholesky
import pylab
from spotPython.design.spacefilling import spacefilling
from spotPython.build.surrogates import surrogates
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from spotPython.utils.repair import repair_non_numeric
from spotPython.utils.aggregate import aggregate_mean_var
import logging
import numpy as np
from typing import List, Union, Tuple, Any, Optional


logger = logging.getLogger(__name__)
# configure the handler and formatter as needed
py_handler = logging.FileHandler(f"{__name__}.log", mode='w')
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
# add formatter to the handler
py_handler.setFormatter(py_formatter)
# add handler to the logger
logger.addHandler(py_handler)


class Kriging(surrogates):
    """Kriging surrogate.

    Attributes:
        nat_range_X (list):
            List of X natural ranges.
        nat_range_y (list):
            List of y nat ranges.
        noise (bool):
            noisy objective function. Default: False. If `True`, regression kriging will be used.
        var_type (str):
            variable type. Can be either `"num`" (numerical) of `"factor"` (factor).
        num_mask (array):
            array of bool variables. `True` represent numerical (float) variables.
        factor_mask (array):
            array of factor variables. `True` represents factor (unordered) variables.
        int_mask (array):
            array of integer variables. `True` represents integers (ordered) variables.
        ordered_mask (array):
            array of ordered variables. `True` represents integers or float (ordered) variables.
            Set of veriables which an order relation, i.e., they are either num (float) or int.
        name (str):
            Surrogate name
        seed (int):
            Random seed.
        use_cod_y (bool):
            Use coded y values.
        sigma (float):
            Kriging sigma.
        gen (method):
            Design generator, e.g., spotPython.design.spacefilling.spacefilling.
        min_theta (float):
            min log10 theta value. Defaults: -6.
        max_theta (float):
            max log10 theta value. Defaults: 3.
        min_p (float):
            min p value. Default: 1.
        max_p (float):
            max p value. Default: 2.
    """
    def __init__(
            self: object,
            noise: bool = False,
            cod_type: Optional[str] = "norm",
            var_type: List[str] = ["num"],
            use_cod_y: bool = False,
            name: str = "kriging",
            seed: int = 124,
            model_optimizer=None,
            model_fun_evals: Optional[int] = None,
            min_theta: float = -3,
            max_theta: float = 2,
            n_theta: int = 1,
            n_p: int = 1,
            optim_p: bool = False,
            log_level: int = 50,
            spot_writer=None,
            counter=None,
            **kwargs
    ):
        """
        Initialize the Kriging surrogate.

        Args:
            noise (bool): Use regression instead of interpolation kriging. Defaults to False.
            cod_type (Optional[str]):
                Normalize or standardize X and values.
                Can be None, "norm", or "std". Defaults to "norm".
            var_type (List[str]):
                Variable type. Can be either "num" (numerical) or "factor" (factor).
                Defaults to ["num"].
            use_cod_y (bool): Use coded y values (instead of natural one). Defaults to False.
            name (str): Surrogate name. Defaults to "kriging".
            seed (int): Random seed. Defaults to 124.
            model_optimizer : Optimizer on the surrogate. If None, differential_evolution is selected.
            model_fun_evals (Optional[int]): Number of iterations used by the optimizer on the surrogate.
            min_theta (float): Min log10 theta value. Defaults to -3.
            max_theta (float): Max log10 theta value. Defaults to 2.
            n_theta (int): Number of theta values. Defaults to 1.
            n_p (int): Number of p values. Defaults to 1.
            optim_p (bool): Determines whether p should be optimized.
            log_level (int): Logging level, e.g., 20 is "INFO". Defaults to 50 ("CRITICAL").
            spot_writer : Spot writer.
            counter : Counter.

        Examples:
            Surrogate of the x*sin(x) function, see [1].

            >>> from spotPython.build.kriging import Kriging
                import numpy as np
                import matplotlib.pyplot as plt
                rng = np.random.RandomState(1)
                X = linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
                y = np.squeeze(X * np.sin(X))
                training_indices = rng.choice(arange(y.size), size=6, replace=False)
                X_train, y_train = X[training_indices], y[training_indices]
                S = Kriging(name='kriging', seed=124)
                S.fit(X_train, y_train)
                mean_prediction, std_prediction = S.predict(X)
                plt.plot(X, y, label=r"$f(x)$", linestyle="dotted")
                plt.scatter(X_train, y_train, label="Observations")
                plt.plot(X, mean_prediction, label="Mean prediction")
                plt.fill_between(
                    X.ravel(),
                    mean_prediction - 1.96 * std_prediction,
                    mean_prediction + 1.96 * std_prediction,
                    alpha=0.5,
                    label=r"95% confidence interval",
                    )
                plt.legend()
                plt.xlabel("$x$")
                plt.ylabel("$f(x)$")
                _ = plt.title("Gaussian process regression on noise-free dataset")
                plt.show()

        References:

            [[1](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html)]
            scikit-learn: Gaussian Processes regression: basic introductory example

        """
        super().__init__(name, seed, log_level)

        self.noise = noise
        self.var_type = var_type
        self.cod_type = cod_type
        self.use_cod_y = use_cod_y
        self.name = name
        self.seed = seed
        self.log_level = log_level
        self.spot_writer = spot_writer
        self.counter = counter

        self.sigma = 0
        self.eps = sqrt(spacing(1))
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.min_p = 1
        self.max_p = 2
        self.min_Lambda = 1e-9
        self.max_Lambda = 1.
        self.n_theta = n_theta
        self.n_p = n_p
        self.optim_p = optim_p
        # Psi matrix condition:
        self.cnd_Psi = 0
        self.inf_Psi = False

        self.model_optimizer = model_optimizer
        if self.model_optimizer is None:
            self.model_optimizer = differential_evolution
        self.model_fun_evals = model_fun_evals
        # differential evaluation uses maxiter = 1000
        # and sets the number of function evaluations to
        # (maxiter + 1) * popsize * N, which results in
        # 1000 * 15 * k, because the default popsize is 15 and
        # N is the number of parameters. This seems to be quite large:
        # for k=2 these are 30 000 iterations. Therefore we set this value to
        # 100
        if self.model_fun_evals is None:
            self.model_fun_evals = 100

        # Logging information
        self.log["negLnLike"] = []
        self.log["theta"] = []
        self.log["p"] = []
        self.log["Lambda"] = []
        # Logger
        logger.setLevel(self.log_level)
        logger.info(f"Starting the logger at level {self.log_level} for module {__name__}:")

    def exp_imp(self, y0: float, s0: float) -> float:
        """
        Calculates the expected improvement for a given function value and error in coded units.

        Args:
            self (object): The Kriging object.
            y0 (float): The function value in coded units.
            s0 (float): The error value.

        Returns:
            float: The expected improvement value.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> S = Kriging(name='kriging', seed=124)
            >>> S.cod_y = [0.0, 0.0, 0.0, 0.0, 0.0]
            >>> S.mean_cod_y = [0.0, 0.0, 0.0, 0.0, 0.0]
            >>> S.exp_imp(1.0, 2.0)
            0.0

        """
        # y_min = min(self.cod_y)
        y_min = min(self.mean_cod_y)
        if s0 <= 0.0:
            EI = 0.0
        elif s0 > 0.0:
            EI_one = (y_min - y0) * (
                    0.5 + 0.5 * erf((1.0 / sqrt(2.0)) * ((y_min - y0) / s0))
            )
            EI_two = (s0 * (1.0 / sqrt(2.0 * pi))) * (
                exp(-(1.0 / 2.0) * ((y_min - y0) ** 2.0 / s0 ** 2.0))
            )
            EI = EI_one + EI_two
        return EI

    def set_de_bounds(self) -> None:
        """
        Determine search bounds for model_optimizer, e.g., differential evolution.

        This method sets the attribute `de_bounds` of the object to a list of lists,
        where each inner list represents the lower and upper bounds for a parameter
        being optimized. The number of inner lists is determined by the number of
        parameters being optimized (`n_theta` and `n_p`), as well as whether noise is
        being considered (`noise`).

        Args:
            self (object): The Kriging object.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> MyClass = Kriging(name='kriging', seed=124)
            >>> obj = MyClass()
            >>> obj.set_de_bounds()
            >>> print(obj.de_bounds)
            [[min_theta, max_theta], [min_theta, max_theta], ..., [min_p, max_p], [min_Lambda, max_Lambda]]

        Returns:
            None
        """
        de_bounds = [[self.min_theta, self.max_theta] for _ in range(self.n_theta)]
        if self.optim_p:
            de_bounds += [[self.min_p, self.max_p] for _ in range(self.n_p)]
            if self.noise:
                de_bounds.append([self.min_Lambda, self.max_Lambda])
        else:
            if self.noise:
                de_bounds.append([self.min_Lambda, self.max_Lambda])
        self.de_bounds = de_bounds

    def extract_from_bounds(self, new_theta_p_Lambda: np.ndarray) -> None:
        """
        Extract `theta`, `p`, and `Lambda` from bounds. The kriging object stores
        `theta` as an array,  `p` as an array, and `Lambda` as a float.

        Args:
            self (object): The Kriging object.
            new_theta_p_Lambda (np.ndarray):
                1d-array with theta, p, and Lambda values. Order is important.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> MyClass = Kriging(name='kriging', seed=124)
            >>> obj = MyClass()
            >>> obj.extract_from_bounds(np.array([1, 2, 3]))
            >>> print(obj.theta)
            [1]
            >>> print(obj.p)
            [2]
            >>> print(obj.Lambda)
            3

        Returns:
            None
        """
        self.theta = new_theta_p_Lambda[:self.n_theta]
        if self.optim_p:
            self.p = new_theta_p_Lambda[self.n_theta:self.n_theta + self.n_p]
            if self.noise:
                self.Lambda = new_theta_p_Lambda[self.n_theta + self.n_p]
        else:
            if self.noise:
                self.Lambda = new_theta_p_Lambda[self.n_theta]

    def optimize_model(self) -> Union[List[float], Tuple[float]]:
        """
        Optimize the model using the specified model_optimizer.

        This method uses the specified model_optimizer to optimize the
        likelihood function (`fun_likelihood`) with respect to the model parameters.
        The optimization is performed within the bounds specified by the attribute
        `de_bounds`.
        The result of the optimization is returned as a list or tuple of optimized parameter values.

        Args:
            self (object): The Kriging object.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> MyClass = Kriging(name='kriging', seed=124)
            >>> obj = MyClass()
            >>> result = obj.optimize_model()
            >>> print(result)
            [optimized_theta, optimized_p, optimized_Lambda]

        Returns:
            result["x"] (Union[List[float], Tuple[float]]):
                A list or tuple of optimized parameter values.
        """
        if self.model_optimizer.__name__ == 'dual_annealing':
            result = self.model_optimizer(func=self.fun_likelihood,
                                          bounds=self.de_bounds)
        elif self.model_optimizer.__name__ == 'differential_evolution':
            result = self.model_optimizer(func=self.fun_likelihood,
                                          bounds=self.de_bounds,
                                          maxiter=self.model_fun_evals,
                                          seed=self.seed)
        elif self.model_optimizer.__name__ == 'direct':
            result = self.model_optimizer(func=self.fun_likelihood,
                                          bounds=self.de_bounds,
                                          # maxfun=self.model_fun_evals,
                                          eps=1e-2)
        elif self.model_optimizer.__name__ == 'shgo':
            result = self.model_optimizer(func=self.fun_likelihood,
                                          bounds=self.de_bounds)
        elif self.model_optimizer.__name__ == 'basinhopping':
            result = self.model_optimizer(func=self.fun_likelihood,
                                          x0=mean(self.de_bounds, axis=1))
        else:
            result = self.model_optimizer(func=self.fun_likelihood, bounds=self.de_bounds)
        return result["x"]

    def update_log(self) -> None:
        """
        Update the log with the current values of negLnLike, theta, p, and Lambda.

        This method appends the current values of negLnLike, theta, p (if optim_p is True),
        and Lambda (if noise is True)
        to their respective lists in the log dictionary.
        It also updates the log_length attribute with the current length
        of the negLnLike list in the log.

        If spot_writer is not None, this method also writes the current values of
        negLnLike, theta, p (if optim_p is True),
        and Lambda (if noise is True) to the spot_writer object.

        Args:
            self (object): The Kriging object.

        Returns:
            None

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> MyClass = Kriging(name='kriging', seed=124)
            >>> obj = MyClass()
            >>> obj.update_log()
            >>> print(obj.log)
            {'negLnLike': [0.5], 'theta': [0.1], 'p': [0.2], 'Lambda': [0.3]}
        """
        self.log["negLnLike"] = append(self.log["negLnLike"], self.negLnLike)
        self.log["theta"] = append(self.log["theta"], self.theta)
        if self.optim_p:
            self.log["p"] = append(self.log["p"], self.p)
        if self.noise:
            self.log["Lambda"] = append(self.log["Lambda"], self.Lambda)
        # get the length of the log
        self.log_length = len(self.log["negLnLike"])
        if self.spot_writer is not None:
            writer = self.spot_writer
            negLnLike = self.negLnLike.copy()
            writer.add_scalar("spot_negLnLike", negLnLike, self.counter+self.log_length)
            # add the self.n_theta theta values to the writer with one key "theta",
            # i.e, the same key for all theta values
            theta = self.theta.copy()
            writer.add_scalars("spot_theta", {f"theta_{i}": theta[i] for i in range(self.n_theta)},
                               self.counter+self.log_length)
            if self.noise:
                Lambda = self.Lambda.copy()
                writer.add_scalar("spot_Lambda", Lambda, self.counter+self.log_length)
            if self.optim_p:
                p = self.p.copy()
                writer.add_scalars("spot_p", {f"p_{i}": p[i] for i in range(self.n_p)}, self.counter+self.log_length)
            writer.flush()

    def fit(self, nat_X: np.ndarray, nat_y: np.ndarray) -> object:
        """
        Fits the hyperparameters (`theta`, `p`, `Lambda`) of the Kriging model.

        The function computes the following internal values:
        1. `theta`, `p`, and `Lambda` values via optimization of the function `fun_likelihood()`.
        2. Correlation matrix `Psi` via `rebuildPsi()`.

        Args:
            self (object): The Kriging object.
            nat_X (np.ndarray): Sample points.
            nat_y (np.ndarray): Function values.

        Returns:
            object: Fitted estimator.

        Attributes:
            theta (np.ndarray): Kriging theta values. Shape (k,).
            p (np.ndarray): Kriging p values. Shape (k,).
            LnDetPsi (np.float64): Determinant Psi matrix.
            Psi (np.matrix): Correlation matrix Psi. Shape (n,n).
            psi (np.ndarray): psi vector. Shape (n,).
            one (np.ndarray): vector of ones. Shape (n,).
            mu (np.float64): Kriging expected mean value mu.
            U (np.matrix): Kriging U matrix, Cholesky decomposition. Shape (n,n).
            SigmaSqr (np.float64): Sigma squared value.
            Lambda (float): lambda noise value.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> nat_X = np.array([[1, 2], [3, 4]])
            >>> nat_y = np.array([1, 2])
            >>> surrogate = Kriging()
            >>> surrogate.fit(nat_X, nat_y)
        """
        self.initialize_variables(nat_X, nat_y)
        self.set_variable_types()
        self.nat_to_cod_init()
        self.set_theta_values()
        self.initialize_matrices()
        # build_Psi() and build_U() are called in fun_likelihood
        self.set_de_bounds()
        # Finally, set new theta and p values and update the surrogate again
        # for new_theta_p_Lambda in de_results["x"]:
        new_theta_p_Lambda = self.optimize_model()
        self.extract_from_bounds(new_theta_p_Lambda)
        self.build_Psi()
        self.build_U()
        # TODO: check if the following line is necessary!
        self.likelihood()
        self.update_log()

    def initialize_variables(self, nat_X: np.ndarray, nat_y: np.ndarray) -> None:
        """
        Initialize variables for the class instance.

        This method takes in the independent and dependent variable data as input
        and initializes the class instance variables.
        It creates deep copies of the input data and stores them in the
        instance variables `nat_X` and `nat_y`.
        It also calculates the number of observations `n` and
        the number of independent variables `k` from the shape of `nat_X`.
        Finally, it creates empty arrays with the same shape as `nat_X`
        and `nat_y` and stores them in the instance variables `cod_X` and `cod_y`.

        Args:
            self (object): The Kriging object.
            nat_X (np.ndarray): The independent variable data.
            nat_y (np.ndarray): The dependent variable data.

        Returns:
            None

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> surrogate = Kriging()
            >>> nat_X = np.array([[1, 2], [3, 4]])
            >>> nat_y = np.array([1, 2])
            >>> surrogate.initialize_variables(nat_X, nat_y)
            >>> surrogate.nat_X
            array([[1, 2],
                     [3, 4]])
            >>> surrogate.nat_y
            array([1, 2])

        """
        self.nat_X = copy.deepcopy(nat_X)
        self.nat_y = copy.deepcopy(nat_y)
        self.n = self.nat_X.shape[0]
        self.k = self.nat_X.shape[1]
        self.cod_X = np.empty_like(self.nat_X)
        self.cod_y = np.empty_like(self.nat_y)

    def set_variable_types(self) -> None:
        """
        Set the variable types for the class instance.

        This method sets the variable types for the class instance based
        on the `var_type` attribute. If the length of `var_type` is less
        than `k`, all variable types are forced to 'num' and a warning is logged.
        The method then creates masks for each variable
        type ('num', 'factor', 'int', 'float') using numpy arrays.

        Args:
            self (object): The Kriging object.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> class MyClass(Kriging):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.var_type = ["num", "factor"]
            >>> instance = MyClass()
            >>> instance.set_variable_types()
            >>> instance.num_mask
            array([ True, False])

        Returns:
            None
        """
        # assume all variable types are "num" if "num" is
        # specified once:
        if len(self.var_type) < self.k:
            self.var_type = self.var_type * self.k
            logger.warning("Warning: All variable types forced to 'num'.")
        self.num_mask = np.array(list(map(lambda x: x == "num", self.var_type)))
        self.factor_mask = np.array(list(map(lambda x: x == "factor", self.var_type)))
        self.int_mask = np.array(list(map(lambda x: x == "int", self.var_type)))
        self.ordered_mask = np.array(list(map(lambda x: x == "int" or x == "num" or x == "float", self.var_type)))

    def set_theta_values(self) -> None:
        """
        Set the theta values for the class instance.

        This method sets the theta values for the class instance based
        on the `n_theta` and `k` attributes. If `n_theta` is greater than
        `k`, `n_theta` is set to `k` and a warning is logged.
        The method then initializes the `theta` attribute as a list
        of zeros with length `n_theta`.
        The `x0_theta` attribute is also initialized as a list of ones
        with length `n_theta`, multiplied by `n / (100 * k)`.

        Args:
            self (object): The Kriging object.
        Returns:
            None

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> class MyClass(Kriging):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.n_theta = 3
            >>>         self.k = 2
            >>> instance = MyClass()
            >>> instance.set_theta_values()
            >>> instance.theta
            array([0., 0., 0.])
        """
        if self.n_theta > self.k:
            self.n_theta = self.k
            logger.warning("More theta values than dimensions. `n_theta` set to `k`.")
        self.theta: List[float] = zeros(self.n_theta)
        # TODO: Currently not used:
        self.x0_theta: List[float] = ones((self.n_theta,)) * self.n / (100 * self.k)

    def initialize_matrices(self) -> None:
        """
        Initialize the matrices for the class instance.

        This method initializes several matrices and attributes for the class instance.
        The `p` attribute is initialized as a list of ones with length `n_p`, multiplied by 2.0.
        The `pen_val` attribute is initialized as the natural logarithm of the
        variance of `nat_y`, multiplied by `n`, plus 1e4.
        The `negLnLike`, `LnDetPsi`, `mu`, `U`, `SigmaSqr`, and `Lambda` attributes are all set to None.
        The `gen` attribute is initialized using the `spacefilling` function with arguments `k` and `seed`.
        The `Psi` attribute is initialized as a zero matrix with shape `(n, n)` and dtype `float64`.
        The `psi` attribute is initialized as a zero matrix with shape `(n, 1)`.
        The `one` attribute is initialized as a list of ones with length `n`.

        Args:
            self (object): The Kriging object.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> class MyClass(Kriging):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.n_p = 2
            >>>         self.n = 3
            >>>         self.nat_y = np.array([1, 2, 3])
            >>>         self.k = 2
            >>>         self.seed = 1
            >>> instance = MyClass()
            >>> instance.initialize_matrices()

        Returns:
            None
        """
        self.p = ones(self.n_p) * 2.0
        self.pen_val = self.n * log(var(self.nat_y)) + 1e4
        self.negLnLike = None
        self.gen = spacefilling(k=self.k, seed=self.seed)
        self.LnDetPsi = None
        self.Psi = zeros((self.n, self.n), dtype=float64)
        self.psi = zeros((self.n, 1))
        self.one = ones(self.n)
        self.mu = None
        self.U = None
        self.SigmaSqr = None
        self.Lambda = None

    def fun_likelihood(self, new_theta_p_Lambda: np.ndarray) -> float:
        """
        Compute log likelihood for a set of hyperparameters (theta, p, Lambda).

        This method computes the log likelihood for a set of hyperparameters
        (theta, p, Lambda) by performing the following steps:
        1. Extracts the hyperparameters from the input array using `extract_from_bounds()`.
        2. Checks if any element in `10^theta` is equal to 0. If so, logs a warning and
        returns the penalty value (`pen_val`).
        3. Builds the `Psi` matrix using `build_Psi()`.
        4. Checks if `Psi` is ill-conditioned or infinite. If so, logs a warning and returns
        the penalty value (`pen_val`).
        5. Builds the `U` matrix using `build_U()`. If an exception occurs, logs an error and
        returns the penalty value (`pen_val`).
        6. Computes the negative log likelihood using `likelihood()`.
        7. Returns the computed negative log likelihood (`negLnLike`).

        Args:
            self (object): The Kriging object.
            new_theta_p_Lambda (np.ndarray):
                An array containing the `theta`, `p`, and `Lambda` values.

        Returns:
            float:
                The negative log likelihood of the surface at the specified hyperparameters.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> class MyClass(Kriging):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.n_p = 2
            >>>         self.n = 3
            >>>         self.nat_y = np.array([1, 2, 3])
            >>>         self.k = 2
            >>>         self.seed = 1
            >>> instance = MyClass()
            >>> negLnLike = instance.fun_likelihood(new_theta_p_Lambda)
            >>> print(negLnLike)

        """
        self.extract_from_bounds(new_theta_p_Lambda)
        if self.__is_any__(power(10.0, self.theta), 0):
            logger.warning("Failure in fun_likelihood: 10^theta == 0. Setting negLnLike to %s", self.pen_val)
            return self.pen_val
        self.build_Psi()
        if (self.inf_Psi or self.cnd_Psi > 1e9):
            logger.warning("Failure in fun_likelihood: Psi is ill conditioned: %s", self.cnd_Psi)
            logger.warning("Setting negLnLike to: %s", self.pen_val)
            return self.pen_val

        try:
            self.build_U()
        except Exception as error:
            penalty_value = self.pen_val
            print("Error in fun_likelihood(). Call to build_U() failed.")
            print("error=%s, type(error)=%s" % (error, type(error)))
            print("Setting negLnLike to %.2f." % self.pen_val)
            return penalty_value
        self.likelihood()
        return self.negLnLike

    def __is_any__(self, x: Union[np.ndarray, Any], v: Any) -> bool:
        """
        Check if any element in `x` is equal to `v`.

        This method checks if any element in the input array `x` is equal to the value `v`.
        If `x` is not an instance of `ndarray`, it is first converted to a numpy array using
        the `array()` function.

        Args:
            self (object): The Kriging object.
            x (np.ndarray or array-like):
                The input array to check for the presence of value `v`.
            v (scalar):
                The value to check for in the input array `x`.

        Returns:
            bool:
                True if any element in `x` is equal to `v`, False otherwise.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> class MyClass(Kriging):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.n_p = 2
            >>>         self.n = 3
            >>>         self.nat_y = np.array([1, 2, 3])
            >>>         self.k = 2
            >>>         self.seed = 1

            >>> instance = MyClass()
            >>> result = instance.__is_any__(x, v)
            >>> print(result)

        """
        if not isinstance(x, ndarray):
            x = array([x])
        return any(x == v)

    def build_Psi(self) -> None:
        """
        Constructs a new (n x n) correlation matrix Psi to reflect new data
        or a change in hyperparameters.

        This method uses `theta`, `p`, and coded `X` values to construct the
        correlation matrix as described in [Forr08a, p.57].

        Args:
            self (object): The Kriging object.

        Returns:
            None

        Raises:
            LinAlgError: If building Psi fails.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> class MyClass(Kriging):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.n_p = 2
            >>>         self.n = 3
            >>>         self.nat_y = np.array([1, 2, 3])
            >>>         self.k = 2
            >>>         self.seed = 1

            >>> obj = MyClass()
            >>> obj.build_Psi()

        """
        self.Psi = zeros((self.n, self.n), dtype=float64)
        theta = power(10.0, self.theta)
        if self.n_theta == 1:
            theta = theta * ones(self.k)
        try:
            D = zeros((self.n, self.n))
            if self.ordered_mask.any():
                X_ordered = self.cod_X[:, self.ordered_mask]
                D = squareform(
                    pdist(
                        X_ordered, metric='sqeuclidean', out=None, w=theta[self.ordered_mask]))
            if self.factor_mask.any():
                X_factor = self.cod_X[:, self.factor_mask]
                D = (D + squareform(
                    pdist(X_factor,
                          metric='hamming',
                          out=None,
                          w=theta[self.factor_mask])))
            self.Psi = exp(-D)
        except LinAlgError as err:
            print(f"Building Psi failed:\n {self.Psi}. {err=}, {type(err)=}")
        if self.noise:
            self.Psi[diag_indices_from(self.Psi)] += self.Lambda
        else:
            self.Psi[diag_indices_from(self.Psi)] += self.eps
        if (isinf(self.Psi)).any():
            self.inf_Psi = True
        self.cnd_Psi = cond(self.Psi)

    def build_U(self, scipy: bool = True) -> None:
        """
        Performs Cholesky factorization of Psi as U as described in [Forr08a, p.57].

        This method uses either `scipy_cholesky` or numpy's `cholesky` to perform the Cholesky factorization of Psi.

        Args:
            self (object):
                The Kriging object.
            scipy (bool):
                If True, use `scipy_cholesky`.
                If False, use numpy's `cholesky`.
                Defaults to True.

        Returns:
            None

        Raises:
            LinAlgError:
                If Cholesky factorization fails for Psi.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> class MyClass(Kriging):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.n_p = 2
            >>>         self.n = 3
            >>>         self.nat_y = np.array([1, 2, 3])
            >>>         self.k = 2
            >>>         self.seed = 1

            >>> obj = MyClass()
            >>> obj.build_U()
        """
        try:
            self.U = scipy_cholesky(self.Psi, lower=True) if scipy else cholesky(self.Psi)
            self.U = self.U.T
        except LinAlgError as err:
            print(f"build_U() Cholesky failed for Psi:\n {self.Psi}. {err=}, {type(err)=}")

    def likelihood(self) -> None:
        """
        Calculates the negative of the concentrated log-likelihood.

        This method implements equation (2.32) in [Forr08a] to calculate
        the negative of the concentrated log-likelihood. It also modifies `mu`,
        `SigmaSqr`, `LnDetPsi`, and `negLnLike`.

        Note:
            `build_Psi` and `build_U` should be called first.

        Args:
            self (object):
                The Kriging object.

        Returns:
            None

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> class MyClass(Kriging):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.n_p = 2
            >>>         self.n = 3
            >>>         self.nat_y = np.array([1, 2, 3])
            >>>         self.k = 2
            >>>         self.seed = 1

            >>> obj = MyClass()
            >>> obj.build_Psi()
            >>> obj.build_U()
            >>> obj.likelihood()
        """
        # (2.20) in [Forr08a]:
        U_T_inv_one = solve(self.U.T, self.one)
        U_T_inv_cod_y = solve(self.U.T, self.cod_y)
        mu = self.one.T.dot(solve(self.U, U_T_inv_cod_y)) / self.one.T.dot(solve(self.U, U_T_inv_one))
        self.mu = mu
        # (2.31) in [Forr08a]
        cod_y_minus_mu = self.cod_y - self.one.dot(self.mu)
        self.SigmaSqr = cod_y_minus_mu.T.dot(solve(self.U, solve(self.U.T, cod_y_minus_mu))) / self.n
        # (2.32) in [Forr08a]
        self.LnDetPsi = 2.0 * sum(log(abs(diag(self.U))))
        self.negLnLike = -1.0 * (-(self.n / 2.0) * log(self.SigmaSqr) - 0.5 * self.LnDetPsi)

    def plot(self, show: Optional[bool] = True) -> None:
        """
        This function plots 1D and 2D surrogates.

        Args:
            self (object):
                The Kriging object.
            show (bool):
                If `True`, the plots are displayed.
                If `False`, `plt.show()` should be called outside this function.

        Returns:
            None

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> class MyClass(Kriging):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.n_p = 2
            >>>         self.n = 3
            >>>         self.nat_y = np.array([1, 2, 3])
            >>>         self.k = 2
            >>>         self.seed = 1

            >>> plot(show=True)
        """
        if self.k == 1:
            # TODO: Improve plot (add conf. interval etc.)
            fig = pylab.figure(figsize=(9, 6))
            # t1 = array(arange(0.0, 1.0, 0.01))
            # y1 = array([self.predict(array([x]), return_val="y") for x in t1])
            # plt.figure()
            # plt.plot(t1, y1, "k")
            # if show:
            #     plt.show()
            #
            n_grid = 100
            x = linspace(
                self.nat_range_X[0][0], self.nat_range_X[0][1], num=n_grid
            )
            y = self.predict(x)
            plt.figure()
            plt.plot(x, y, "k")
            if show:
                plt.show()

        if self.k == 2:
            fig = pylab.figure(figsize=(9, 6))
            n_grid = 100
            x = linspace(
                self.nat_range_X[0][0], self.nat_range_X[0][1], num=n_grid
            )
            y = linspace(
                self.nat_range_X[1][0], self.nat_range_X[1][1], num=n_grid
            )
            X, Y = meshgrid(x, y)
            # Predict based on the optimized results
            zz = array(
                [self.predict(array([x, y]), return_val="all") for x, y in zip(ravel(X), ravel(Y))]
            )
            zs = zz[:, 0, :]
            zse = zz[:, 1, :]
            Z = zs.reshape(X.shape)
            Ze = zse.reshape(X.shape)

            if self.cod_type == "norm":
                nat_point_X = (
                                      self.cod_X[:, 0] * (self.nat_range_X[0][1] - self.nat_range_X[0][0])
                              ) + self.nat_range_X[0][0]
                nat_point_Y = (
                                      self.cod_X[:, 1] * (self.nat_range_X[1][1] - self.nat_range_X[1][0])
                              ) + self.nat_range_X[1][0]
            elif self.cod_type == "std":
                nat_point_X = self.cod_X[:, 0] * self.nat_std_X[0] + self.nat_mean_X[0]
                nat_point_Y = self.cod_X[:, 1] * self.nat_std_X[1] + self.nat_mean_X[1]
            else:
                nat_point_X = self.cod_X[:, 0]
                nat_point_Y = self.cod_X[:, 1]
            contour_levels = 30
            ax = fig.add_subplot(224)
            # plot predicted values:
            pylab.contourf(X, Y, Ze, contour_levels, cmap="jet")
            pylab.title("Error")
            pylab.colorbar()
            # plot observed points:
            pylab.plot(nat_point_X, nat_point_Y, "ow")
            #
            ax = fig.add_subplot(223)
            # plot predicted values:
            plt.contourf(X, Y, Z, contour_levels, zorder=1, cmap="jet")
            plt.title("Surrogate")
            # plot observed points:
            pylab.plot(nat_point_X, nat_point_Y, "ow", zorder=3)
            pylab.colorbar()
            #
            ax = fig.add_subplot(221, projection="3d")
            ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.9, cmap="jet")
            #
            ax = fig.add_subplot(222, projection="3d")
            ax.plot_surface(X, Y, Ze, rstride=3, cstride=3, alpha=0.9, cmap="jet")
            #
            pylab.show()

    def predict(self, nat_X: ndarray, nat: bool = True, return_val: str = "y") -> Union[float,
                                                                                        Tuple[float,
                                                                                              float,
                                                                                              float]]:
        """
        This function returns the prediction (in natural units) of the surrogate at the natural coordinates of X.

        Args:
            self (object):
                The Kriging object.
            nat_X (ndarray):
                Design variable to evaluate in natural units.
            nat (bool):
                argument `nat_X` is in natural range. Default: `True`.
                If set to `False`, `nat_X` will not be normalized (which might be useful
                if already normalized y values are used).
            return_val (str):
                whether `y`, `s`, neg. `ei` (negative expected improvement),
                or all three values are returned.
                Default is (for compatibility with sklearn) "y". To return `s`, select "s",
                to return neg. `ei`, select "ei".
                To return the tuple `(y, s, ei)`, select "all".

        Returns:
            float:
                The predicted value in natural units if return_val is "y".
            float:
                predicted error if return_val is "s".
            float:
                expected improvement if return_val is "ei".
            Tuple[float, float, float]:
                The predicted value in natural units, predicted error
                and expected improvement if return_val is "all".

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> from numpy import array
            >>> X = array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
            >>> y = array([0.0, 0.01, 0.04])
            >>> k = Kriging(X, y)
            >>> k.predict(array([[0.3, 0.3]]))
            array([0.09])

        """
        # Check for the shape and the type of the Input
        if isinstance(nat_X, ndarray):
            try:
                X = nat_X.reshape(-1, self.nat_X.shape[1])
                X = repair_non_numeric(X, self.var_type)
            except Exception:
                raise TypeError("13.1: Input to predict was not convertible to the size of X")
        else:
            raise TypeError(f"type of the given input is an {type(nat_X)} instead of an ndarray")
        n = X.shape[0]
        y = empty(n, dtype=float)
        s = empty(n, dtype=float)
        ei = empty(n, dtype=float)
        for i in range(n):
            if nat:
                x = self.nat_to_cod_x(X[i, :])
            else:
                x = X[i, :]
            y[i], s[i], ei[i] = self.predict_coded(x)
        if return_val == "y":
            return y
        elif return_val == "s":
            return s
        elif return_val == "ei":
            return -1.0 * ei
        else:
            return y, s, -1.0 * ei

    def build_psi_vec(self, cod_x: ndarray) -> None:
        """
        Build the psi vector. Needed by `predict_cod`, `predict_err_coded`,
        `regression_predict_coded`. Modifies `self.psi`.

        Args:
            self (object):
                The Kriging object.
            cod_x (ndarray):
                point to calculate psi

        Returns:
            None

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> from numpy import array
            >>> X = array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
            >>> y = array([0.0, 0.01, 0.04])
            >>> k = Kriging(X, y)
            >>> cod_x = array([0.3, 0.3])
            >>> build_psi_vec(cod_x)

        """
        self.psi = zeros((self.n))
        # theta = self.theta  # TODO:
        theta = power(10.0, self.theta)
        if self.n_theta == 1:
            theta = theta * ones(self.k)
        try:
            D = zeros((self.n))
            if self.ordered_mask.any():
                X_ordered = self.cod_X[:, self.ordered_mask]
                x_ordered = cod_x[self.ordered_mask]
                D = cdist(x_ordered.reshape(-1, sum(self.ordered_mask)),
                          X_ordered.reshape(-1, sum(self.ordered_mask)),
                          metric='sqeuclidean',
                          out=None,
                          w=theta[self.ordered_mask])
            if self.factor_mask.any():
                X_factor = self.cod_X[:, self.factor_mask]
                x_factor = cod_x[self.factor_mask]
                D = (D + cdist(x_factor.reshape(-1, sum(self.factor_mask)),
                               X_factor.reshape(-1, sum(self.factor_mask)),
                               metric='hamming',
                               out=None,
                               w=theta[self.factor_mask]))
            self.psi = exp(-D).T
        except LinAlgError as err:
            print(f"Building psi failed:\n {self.psi}. {err=}, {type(err)=}")

    def predict_coded(self, cod_x: np.ndarray) -> Tuple[float, float, float]:
        """
        Kriging prediction of one point in the coded units as described in (2.20) in [Forr08a].
        The error is returned as well.

        Args:
            self (object):
                The Kriging object.
            cod_x (np.ndarray):
                Point in coded units to make prediction at.

        Returns:
            f (float): Predicted value in coded units.
            SSqr (float): Predicted error.
            EI (float): Expected improvement.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> from numpy import array
            >>> X = array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
            >>> y = array([0.0, 0.01, 0.04])
            >>> k = Kriging(X, y)
            >>> cod_x = array([0.3, 0.3])
            >>> k.predict_coded(cod_x)
            (0.09, 0.0, 0.0)

        Note:
            `self.mu` and `self.SigmaSqr` are computed in `likelihood`, not here.
            See also [Forr08a, p.60].
        """
        self.build_psi_vec(cod_x)
        U_T_inv = solve(self.U.T, self.cod_y - self.one.dot(self.mu))
        f = self.mu + self.psi.T.dot(solve(self.U, U_T_inv))
        if self.noise:
            Lambda = self.Lambda
        else:
            Lambda = 0.0
        # Error in [Forr08a, p.87]:
        SSqr = self.SigmaSqr * (1 + Lambda - self.psi.T.dot(solve(self.U, solve(self.U.T, self.psi))))
        SSqr = power(abs(SSqr[0]), 0.5)[0]
        EI = self.exp_imp(y0=f[0], s0=SSqr)
        return f[0], SSqr, EI

    def weighted_exp_imp(self, cod_x: np.ndarray, w: float) -> float:
        """
        Weighted expected improvement.

        Args:
            self (object): The Kriging object.
            cod_x (np.ndarray): A coded design vector.
            w (float): Weight.

        Returns:
            EI (float): Weighted expected improvement.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> from numpy import array
            >>> X = array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
            >>> y = array([0.0, 0.01, 0.04])
            >>> k = Kriging(X, y)
            >>> cod_x = array([0.3, 0.3])
            >>> w = 0.5
            >>> k.weighted_exp_imp(cod_x, w)
            0.0

        References:

            [Sobester et al. 2005].
        """
        y0, s0 = self.predict_coded(cod_x)
        y_min = min(self.cod_y)
        if s0 <= 0.0:
            EI = 0.0
        else:
            y_min_y0 = y_min - y0
            EI_one = w * (
                    y_min_y0
                    * (0.5 + 0.5 * erf((1.0 / sqrt(2.0)) * (y_min_y0 / s0)))
            )
            EI_two = (
                    (1.0 - w)
                    * (s0 * (1.0 / sqrt(2.0 * pi)))
                    * (exp(-(1.0 / 2.0) * ((y_min_y0) ** 2.0 / s0 ** 2.0)))
            )
            EI = EI_one + EI_two
        return EI

    def calculate_mean_MSE(self, n_samples: int = 200, points: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Calculates the mean MSE metric of the model by evaluating MSE at a number of points.

        Args:
            self (object):
                The Kriging object.
            n_samples (int):
                Number of points to sample the mean squared error at.
                Ignored if the points argument is specified.
            points (np.ndarray):
                An array of points to sample the model at.

        Returns:
            mean_MSE (float): The mean value of MSE.
            std_MSE (float): The standard deviation of the MSE points.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> from numpy import array
            >>> X = array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
            >>> y = array([0.0, 0.01, 0.04])
            >>> k = Kriging(X, y)
            >>> n_samples = 200
            >>> mean_MSE, std_MSE = k.calculate_mean_MSE(n_samples)
            >>> print(f"Mean MSE: {mean_MSE}, Standard deviation of MSE: {std_MSE}")

        """
        if points is None:
            points = self.gen.lhd(n_samples)
        values = [self.predict(cod_X=point, nat=True, return_val="s") for point in points]
        return mean(values), std(values)

    def cod_to_nat_x(self, cod_X: np.ndarray) -> np.ndarray:
        """
        Converts an array representing one point in normalized (coded) units to natural (physical or real world) units.

        Args:
            self (object): The Kriging object.
            cod_X (np.ndarray):
                An array representing one point (self.k long) in normalized (coded) units.

        Returns:
            X (np.ndarray): An array of natural (physical or real world) units.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> from numpy import array
            >>> X = array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
            >>> y = array([0.0, 0.01, 0.04])
            >>> k = Kriging(X, y)
            >>> cod_X = array([0.3, 0.3])
            >>> nat_X = k.cod_to_nat_x(cod_X)
            >>> print(f"Natural units: {nat_X}")

        """
        X = copy.deepcopy(cod_X)
        if self.cod_type == "norm":
            for i in range(self.k):
                X[i] = (
                    X[i] * float(self.nat_range_X[i][1] - self.nat_range_X[i][0])
                ) + self.nat_range_X[i][0]
            return X
        elif self.cod_type == "std":
            for i in range(self.k):
                X[i] = X[i] * self.nat_std_X[i] + self.nat_mean_X[i]
            return X
        else:
            return cod_X

    def cod_to_nat_y(self, cod_y: np.ndarray) -> np.ndarray:
        """
        Converts a normalized array of coded (model) units in the range of [0,1]
        to an array of observed values in real-world units.

        Args:
            self (object): The Kriging object.
            cod_y (np.ndarray):
                A normalized array of coded (model) units in the range of [0,1].

        Returns:
            y (np.ndarray): An array of observed values in real-world units.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> from numpy import array
            >>> X = array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
            >>> y = array([0.0, 0.01, 0.04])
            >>> k = Kriging(X, y)
            >>> cod_y = array([0.5, 0.5])
            >>> nat_y = k.cod_to_nat_y(cod_y)
            >>> print(f"Real-world units: {nat_y}")

        """
        return (
            cod_y * (self.nat_range_y[1] - self.nat_range_y[0]) + self.nat_range_y[0]
            if self.cod_type == "norm"
            else cod_y * self.nat_std_y + self.nat_mean_y
            if self.cod_type == "std"
            else cod_y
        )

    def nat_to_cod_x(self, nat_X: np.ndarray) -> np.ndarray:
        """
        Normalizes one point (row) of nat_X array to [0,1]. The internal nat_range_X values are not updated.

        Args:
            self (object): The Kriging object.
            nat_X (np.ndarray):
                An array representing one point (self.k long) in natural (physical or real world) units.

        Returns:
            X (np.ndarray): An array of coded values in the range of [0,1] for each dimension.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> from numpy import array
            >>> X = array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
            >>> y = array([0.0, 0.01, 0.04])
            >>> k = Kriging(X, y)
            >>> nat_X = array([5.0, 5.0])
            >>> cod_X = k.nat_to_cod_x(nat_X)
            >>> print(f"Coded values: {cod_X}")

        """
        X = copy.deepcopy(nat_X)
        if self.cod_type == "norm":
            for i in range(self.k):
                # TODO: Check Implementation of range correction if range == 0:
                # rangex <- xmax - xmin
                # rangey <- ymax - ymin
                # xmin[rangex == 0] <- xmin[rangex == 0] - 0.5
                # xmax[rangex == 0] <- xmax[rangex == 0] + 0.5
                # rangex[rangex == 0] <- 1
                # logger.debug(f"self.nat_range_X[{i}]:\n {self.nat_range_X[i]}")
                # logger.debug(f"X[{i}]:\n {X[i]}")
                rangex = float(self.nat_range_X[i][1] - self.nat_range_X[i][0])
                if rangex == 0:
                    self.nat_range_X[i][0] = self.nat_range_X[i][0] - 0.5
                    self.nat_range_X[i][1] = self.nat_range_X[i][1] + 0.5
                X[i] = (X[i] - self.nat_range_X[i][0]) / float(
                    self.nat_range_X[i][1] - self.nat_range_X[i][0]
                )
            return X
        elif self.cod_type == "std":
            for i in range(self.k):
                X[i] = (X[i] - self.nat_mean_X[i]) / self.nat_std_X[i]
            return X
        else:
            return nat_X

    def nat_to_cod_y(self, nat_y: np.ndarray) -> np.ndarray:
        """
        Normalizes natural y values to [0,1].

        Args:
            self (object): The Kriging object.
            nat_y (np.ndarray):
                An array of observed values in natural (real-world) units.

        Returns:
            y (np.ndarray):
                A normalized array of coded (model) units in the range of [0,1].

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> import numpy as np
            >>> kriging = Kriging()
            >>> nat_y = np.array([5.0, 5.0])
            >>> cod_y = kriging.nat_to_cod_y(nat_y)
            >>> print(f"Coded values: {cod_y}")
        """
        return (
            (nat_y - self.nat_range_y[0]) / (self.nat_range_y[1] - self.nat_range_y[0])
            if self.use_cod_y and self.cod_type == "norm"
            else (nat_y - self.nat_mean_y) / self.nat_std_y
            if self.use_cod_y and self.cod_type == "std"
            else nat_y
        )

    def nat_to_cod_init(self) -> None:
        """
        Determines max and min of each dimension and normalizes that axis to a range of [0,1].
        Called when 1) surrogate is initialized and 2) new points arrive, i.e.,
        suggested by the surrogate as infill points.
        This method calls `nat_to_cod_x` and `nat_to_cod_y` and updates the ranges `nat_range_X` and `nat_range_y`.

        Args:
            self (object): The Kriging object.

        Examples:

            >>> from spotPython.build.kriging import Kriging
            >>> kriging = Kriging()
            >>> kriging.nat_to_cod_init()
        """
        self.nat_range_X = []
        self.nat_range_y = []
        for i in range(self.k):
            self.nat_range_X.append([min(self.nat_X[:, i]), max(self.nat_X[:, i])])
        self.nat_range_y.append(min(self.nat_y))
        self.nat_range_y.append(max(self.nat_y))
        self.nat_mean_X = mean(self.nat_X, axis=0)
        self.nat_std_X = std(self.nat_X, axis=0)
        self.nat_mean_y = mean(self.nat_y)
        self.nat_std_y = std(self.nat_y)
        Z = aggregate_mean_var(X=self.nat_X, y=self.nat_y)
        mu = Z[1]
        self.mean_cod_y = empty_like(mu)

        for i in range(self.n):
            self.cod_X[i] = self.nat_to_cod_x(self.nat_X[i])
        for i in range(self.n):
            self.cod_y[i] = self.nat_to_cod_y(self.nat_y[i])
        for i in range(mu.shape[0]):
            self.mean_cod_y[i] = self.nat_to_cod_y(mu[i])
