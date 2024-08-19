import copy
from math import erf
import matplotlib.pyplot as plt
from numpy import max, min, var, mean
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
from numpy import spacing
from numpy import float64
from numpy import append, ndarray, isinf, linspace, meshgrid, ravel, diag_indices_from, empty
from numpy.linalg import cholesky, solve, LinAlgError, cond
from scipy.optimize import differential_evolution
from scipy.linalg import cholesky as scipy_cholesky
import pylab
from spotpython.design.spacefilling import spacefilling
from spotpython.build.surrogates import surrogates
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from spotpython.utils.repair import repair_non_numeric
from spotpython.utils.aggregate import aggregate_mean_var
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
    """
    def __init__(
            self: object,
            noise: bool = False,
            var_type: List[str] = ["num"],
            name: str = "kriging",
            seed: int = 124,
            model_optimizer=None,
            model_fun_evals: Optional[int] = None,
            min_theta: float = -3.0,
            max_theta: float = 2.0,
            n_theta: int = 1,
            theta_init_zero: bool = True,
            p_val: float = 2.0,
            n_p: int = 1,
            optim_p: bool = False,
            min_Lambda: float = 1e-9,
            max_Lambda: float = 1.,
            log_level: int = 50,
            spot_writer=None,
            counter=None,
            metric_factorial="canberra",
            **kwargs
    ):
        """
        Initialize the Kriging surrogate.

        Args:
            noise (bool): Use regression instead of interpolation kriging. Defaults to False.
            var_type (List[str]):
                Variable type. Can be either "num" (numerical) or "factor" (factor).
                Defaults to ["num"].
            name (str):
                Surrogate name. Defaults to "kriging".
            seed (int):
                Random seed. Defaults to 124.
            model_optimizer (Optional[object]):
                Optimizer on the surrogate. If None, differential_evolution is selected.
            model_fun_evals (Optional[int]):
                Number of iterations used by the optimizer on the surrogate.
            min_theta (float):
                Min log10 theta value. Defaults to -3.
            max_theta (float):
                Max log10 theta value. Defaults to 2.
            n_theta (int):
                Number of theta values. Defaults to 1.
            theta_init_zero (bool):
                Initialize theta with zero. Defaults to True.
            p_val (float):
                p value. Used as an initial value if optim_p = True. Otherwise as a constant. Defaults to 2.
            n_p (int):
                Number of p values. Defaults to 1.
            optim_p (bool):
                Determines whether p should be optimized. Deafults to False.
            min_Lambda (float):
                Min Lambda value. Defaults to 1e-9.
            max_Lambda (float):
                Max Lambda value. Defaults to 1.
            log_level (int):
                Logging level, e.g., 20 is "INFO". Defaults to 50 ("CRITICAL").
            spot_writer (Optional[object]):
                Spot writer. Defaults to None.
            counter (Optional[int]):
                Counter. Defaults to None.
            metric_factorial (str):
                Metric for factorial. Defaults to "canberra". Can be "euclidean",
                "cityblock", seuclidean", "sqeuclidean", "cosine",
                "correlation", "hamming", "jaccard", "jensenshannon",
                "chebyshev", "canberra", "braycurtis", "mahalanobis", "matching".

        Examples:
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                import matplotlib.pyplot as plt
                from numpy import linspace, arange
                rng = np.random.RandomState(1)
                X = linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
                y = np.squeeze(X * np.sin(X))
                training_indices = rng.choice(arange(y.size), size=6, replace=False)
                X_train, y_train = X[training_indices], y[training_indices]
                S = Kriging(name='kriging', seed=124)
                S.fit(X_train, y_train)
                mean_prediction, std_prediction, s_ei = S.predict(X, return_val="all")
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
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
            [[1](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html)]
            scikit-learn: Gaussian Processes regression: basic introductory example

        """
        super().__init__(name, seed, log_level)

        self.noise = noise
        self.var_type = var_type
        self.name = name
        self.seed = seed
        self.log_level = log_level
        self.spot_writer = spot_writer
        self.counter = counter
        self.metric_factorial = metric_factorial

        self.sigma = 0
        self.eps = sqrt(spacing(1))
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.min_p = 1
        self.max_p = 2
        self.min_Lambda = min_Lambda
        self.max_Lambda = max_Lambda
        self.n_theta = n_theta
        self.p_val = p_val
        self.n_p = n_p
        self.optim_p = optim_p
        self.theta_init_zero = theta_init_zero
        # Psi matrix condition:
        self.cnd_Psi = 0
        self.inf_Psi = False

        self.model_optimizer = model_optimizer
        if self.model_optimizer is None:
            self.model_optimizer = differential_evolution
        self.model_fun_evals = model_fun_evals
        # differential evolution uses maxiter = 1000
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
            >>> from spotpython.build.kriging import Kriging
                S = Kriging(name='kriging', seed=124)
                S.aggregated_mean_y = [0.0, 0.0, 0.0, 0.0, 0.0]
                S.exp_imp(1.0, 0.0)
                0.0
            >>> from spotpython.build.kriging import Kriging
                S = Kriging(name='kriging', seed=124)
                S.aggregated_mean_y = [0.0, 0.0, 0.0, 0.0, 0.0]
                # assert S.exp_imp(0.0, 1.0) == 1/np.sqrt(2*np.pi)
                # which is approx. 0.3989422804014327
                S.exp_imp(0.0, 1.0)
                0.3989422804014327
        """
        # We do not use the min y values, but the aggragated mean values
        # y_min = min(self.nat_y)
        y_min = min(self.aggregated_mean_y)
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
            >>> from spotpython.build.kriging import Kriging
                S = Kriging(name='kriging', seed=124)
                S.set_de_bounds()
                print(S.de_bounds)

        Returns:
            None
        """
        logger.debug("In set_de_bounds(): self.min_theta: %s", self.min_theta)
        logger.debug("In set_de_bounds(): self.max_theta: %s", self.max_theta)
        logger.debug("In set_de_bounds(): self.n_theta: %s", self.n_theta)
        logger.debug("In set_de_bounds(): self.optim_p: %s", self.optim_p)
        logger.debug("In set_de_bounds(): self.min_p: %s", self.min_p)
        logger.debug("In set_de_bounds(): self.max_p: %s", self.max_p)
        logger.debug("In set_de_bounds(): self.n_p: %s", self.n_p)
        logger.debug("In set_de_bounds(): self.noise: %s", self.noise)
        logger.debug("In set_de_bounds(): self.min_Lambda: %s", self.min_Lambda)
        logger.debug("In set_de_bounds(): self.max_Lambda: %s", self.max_Lambda)

        de_bounds = [[self.min_theta, self.max_theta] for _ in range(self.n_theta)]
        if self.optim_p:
            de_bounds += [[self.min_p, self.max_p] for _ in range(self.n_p)]
            if self.noise:
                de_bounds.append([self.min_Lambda, self.max_Lambda])
        else:
            if self.noise:
                de_bounds.append([self.min_Lambda, self.max_Lambda])
        self.de_bounds = de_bounds
        logger.debug("In set_de_bounds(): self.de_bounds: %s", self.de_bounds)

    def extract_from_bounds(self, new_theta_p_Lambda: np.ndarray) -> None:
        """
        Extract `theta`, `p`, and `Lambda` from bounds. The kriging object stores
        `theta` as an array,  `p` as an array, and `Lambda` as a float.

        Args:
            self (object): The Kriging object.
            new_theta_p_Lambda (np.ndarray):
                1d-array with theta, p, and Lambda values. Order is important.

        Examples:
            >>> import numpy as np
                from spotpython.build.kriging import Kriging
                n=2
                p=4
                S = Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=False)
                S.extract_from_bounds(np.array([1, 2, 3]))
                print(S.theta)
                print(S.p)
                [1 2]
                [3]

        Returns:
            None
        """
        logger.debug("In extract_from_bounds(): new_theta_p_Lambda: %s", new_theta_p_Lambda)
        self.theta = new_theta_p_Lambda[:self.n_theta]
        logger.debug("In extract_from_bounds(): self.n_theta: %s", self.n_theta)
        if self.optim_p:
            self.p = new_theta_p_Lambda[self.n_theta:self.n_theta + self.n_p]
            logger.debug("In extract_from_bounds(): self.p: %s", self.p)
            if self.noise:
                self.Lambda = new_theta_p_Lambda[self.n_theta + self.n_p]
                logger.debug("In extract_from_bounds(): self.Lambda: %s", self.Lambda)
        else:
            if self.noise:
                self.Lambda = new_theta_p_Lambda[self.n_theta]
                logger.debug("In extract_from_bounds(): self.Lambda: %s", self.Lambda)

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
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                nat_X = np.array([[1, 2], [3, 4]])
                nat_y = np.array([1, 2])
                n=2
                p=2
                S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
                S.initialize_variables(nat_X, nat_y)
                S.set_variable_types()
                S.set_theta_values()
                S.initialize_matrices()
                S.set_de_bounds()
                new_theta_p_Lambda = S.optimize_model()
                print(new_theta_p_Lambda)

        Returns:
            result["x"] (Union[List[float], Tuple[float]]):
                A list or tuple of optimized parameter values.
        """
        logger.debug("In optimize_model(): self.de_bounds passed to optimizer: %s", self.de_bounds)
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
        logger.debug("In optimize_model(): result: %s", result)
        logger.debug('In optimize_model(): returned result["x"]: %s', result["x"])
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
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                nat_X = np.array([[1, 2], [3, 4]])
                nat_y = np.array([1, 2])
                n=2
                p=2
                S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
                S.initialize_variables(nat_X, nat_y)
                S.set_variable_types()
                S.set_theta_values()
                S.initialize_matrices()
                S.set_de_bounds()
                new_theta_p_Lambda = S.optimize_model()
                S.update_log()
                print(S.log)
                {'negLnLike': array([-1.38629436]),
                 'theta': array([-1.14525993,  1.6123372 ]),
                  'p': array([1.84444406, 1.74590865]),
                  'Lambda': array([0.44268472])}

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
            negLnLike = self.negLnLike.copy()
            self.spot_writer.add_scalar("spot_negLnLike", negLnLike, self.counter+self.log_length)
            # add the self.n_theta theta values to the writer with one key "theta",
            # i.e, the same key for all theta values
            theta = self.theta.copy()
            self.spot_writer.add_scalars("spot_theta", {f"theta_{i}": theta[i] for i in range(self.n_theta)},
                                         self.counter+self.log_length)
            if self.noise:
                Lambda = self.Lambda.copy()
                self.spot_writer.add_scalar("spot_Lambda", Lambda, self.counter+self.log_length)
            if self.optim_p:
                p = self.p.copy()
                self.spot_writer.add_scalars("spot_p", {f"p_{i}": p[i] for i in range(self.n_p)}, self.counter+self.log_length)
            self.spot_writer.flush()

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
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                nat_X = np.array([[1, 0], [1, 0]])
                nat_y = np.array([1, 2])
                S = Kriging()
                S.fit(nat_X, nat_y)
                print(S.Psi)
                [[1.00000001 1.        ]
                [1.         1.00000001]]

        """
        logger.debug("In fit(): nat_X: %s", nat_X)
        logger.debug("In fit(): nat_y: %s", nat_y)
        self.initialize_variables(nat_X, nat_y)
        self.set_variable_types()
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
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                nat_X = np.array([[1, 2], [3, 4]])
                nat_y = np.array([1, 2])
                S = Kriging()
                S.initialize_variables(nat_X, nat_y)
                print(f"S.nat_X: {S.nat_X}")
                print(f"S.nat_y: {S.nat_y}")
                S.nat_X: [[1 2]
                          [3 4]]
                S.nat_y: [1 2]

        """
        self.nat_X = copy.deepcopy(nat_X)
        self.nat_y = copy.deepcopy(nat_y)
        self.n = self.nat_X.shape[0]
        self.k = self.nat_X.shape[1]

        self.min_X = min(self.nat_X, axis=0)
        self.max_X = max(self.nat_X, axis=0)

        Z = aggregate_mean_var(X=self.nat_X, y=self.nat_y)
        # aggregated y values:
        mu = Z[1]
        self.aggregated_mean_y = np.copy(mu)
        logger.debug("In initialize_variables(): self.nat_X: %s", self.nat_X)
        logger.debug("In initialize_variables(): self.nat_y: %s", self.nat_y)
        logger.debug("In initialize_variables(): self.aggregated_mean_y: %s", self.aggregated_mean_y)
        logger.debug("In initialize_variables(): self.min_X: %s", self.min_X)
        logger.debug("In initialize_variables(): self.max_X: %s", self.max_X)
        logger.debug("In initialize_variables(): self.n: %s", self.n)
        logger.debug("In initialize_variables(): self.k: %s", self.k)

    def set_variable_types(self) -> None:
        """
        Set the variable types for the class instance.

        This method sets the variable types for the class instance based
        on the `var_type` attribute. If the length of `var_type` is less
        than `k`, all variable types are forced to 'num' and a warning is logged.
        The method then creates Boolean masks for each variable
        type ('num', 'factor', 'int', 'ordered') using numpy arrays, e.g.,
        `num_mask = array([ True,  True])` if two numerical variables are present.

        Args:
            self (object): The Kriging object.

        Examples:
            >>> from spotpython.build.kriging import Kriging
                nat_X = np.array([[1, 2], [3, 4]])
                nat_y = np.array([1, 2])
                n=2
                p=2
                S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
                S.initialize_variables(nat_X, nat_y)
                S.set_variable_types()
                assert S.var_type == ['num', 'num']
                assert S.var_type == ['num', 'num']
                assert S.num_mask.all() == True
                assert S.factor_mask.all() == False
                assert S.int_mask.all() == False
                assert S.ordered_mask.all() == True

        Returns:
            None
        """
        logger.debug("In set_variable_types(): self.k: %s", self.k)
        logger.debug("In set_variable_types(): self.var_type: %s", self.var_type)
        # assume all variable types are "num" if "num" is
        # specified once:
        if len(self.var_type) < self.k:
            self.var_type = self.var_type * self.k
            logger.warning("In set_variable_types(): All variable types forced to 'num'.")
            logger.debug("In set_variable_types(): self.var_type: %s", self.var_type)
        self.num_mask = np.array(list(map(lambda x: x == "num", self.var_type)))
        self.factor_mask = np.array(list(map(lambda x: x == "factor", self.var_type)))
        self.int_mask = np.array(list(map(lambda x: x == "int", self.var_type)))
        self.ordered_mask = np.array(list(map(lambda x: x == "int" or x == "num" or x == "float", self.var_type)))
        logger.debug("In set_variable_types(): self.num_mask: %s", self.num_mask)
        logger.debug("In set_variable_types(): self.factor_mask: %s", self.factor_mask)
        logger.debug("In set_variable_types(): self.int_mask: %s", self.int_mask)
        logger.debug("In set_variable_types(): self.ordered_mask: %s", self.ordered_mask)

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
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                from numpy import array
                nat_X = np.array([[1, 2], [3, 4]])
                nat_y = np.array([1, 2])
                n=2
                p=2
                S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
                S.initialize_variables(nat_X, nat_y)
                S.set_variable_types()
                S.set_theta_values()
                assert S.theta.all() == array([0., 0.]).all()
        """
        logger.debug("In set_theta_values(): self.k: %s", self.k)
        logger.debug("In set_theta_values(): self.n_theta: %s", self.n_theta)
        if ((self.n_theta > 1) or (self.n_theta > self.k)) and (self.n_theta != self.k):
            self.n_theta = self.k
            logger.warning("Too few theta values or more theta values than dimensions. `n_theta` set to `k`.")
            logger.debug("In set_theta_values(): self.n_theta: %s", self.n_theta)
        if self.theta_init_zero:
            self.theta: List[float] = zeros(self.n_theta)
            logger.debug("In set_theta_values(): self.theta: %s", self.theta)
        else:
            logger.debug("In set_theta_values(): self.n: %s", self.n)
            self.theta: List[float] = ones((self.n_theta,)) * self.n / (100 * self.k)
            logger.debug("In set_theta_values(): self.theta: %s", self.theta)

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
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                from numpy import log, var
                nat_X = np.array([[1, 2], [3, 4], [5, 6]])
                nat_y = np.array([1, 2, 3])
                n=3
                p=1
                S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
                S.initialize_variables(nat_X, nat_y)
                S.set_variable_types()
                S.set_theta_values()
                S.initialize_matrices()
                # if var(self.nat_y) is > 0, then self.pen_val = self.n * log(var(self.nat_y)) + 1e4
                # else self.pen_val = self.n * var(self.nat_y) + 1e4
                assert S.pen_val == nat_X.shape[0] * log(var(S.nat_y)) + 1e4
                assert S.Psi.shape == (n, n)

        Returns:
            None
        """
        logger.debug("In initialize_matrices(): self.n_p: %s", self.n_p)
        self.p = ones(self.n_p) * self.p_val
        logger.debug("In initialize_matrices(): self.p: %s", self.p)
        # if var(self.nat_y) is > 0, then self.pen_val = self.n * log(var(self.nat_y)) + 1e4
        # else self.pen_val = self.n * var(self.nat_y) + 1e4
        logger.debug("In initialize_matrices(): var(self.nat_y): %s", var(self.nat_y))
        logger.debug("In initialize_matrices(): self.n: %s", self.n)
        if var(self.nat_y) > 0:
            self.pen_val = self.n * log(var(self.nat_y)) + 1e4
        else:
            self.pen_val = self.n * var(self.nat_y) + 1e4
        logger.debug("In initialize_matrices(): self.pen_val: %s", self.pen_val)
        self.negLnLike = None
        logger.debug("In initialize_matrices(): self.k: %s", self.k)
        logger.debug("In initialize_matrices(): self.seed: %s", self.seed)
        self.gen = spacefilling(k=self.k, seed=self.seed)
        logger.debug("In initialize_matrices(): self.gen: %s", self.gen)
        self.LnDetPsi = None
        self.Psi = zeros((self.n, self.n), dtype=float64)
        logger.debug("In initialize_matrices(): self.Psi: %s", self.Psi)
        self.psi = zeros((self.n, 1))
        logger.debug("In initialize_matrices(): self.psi: %s", self.psi)
        self.one = ones(self.n)
        logger.debug("In initialize_matrices(): self.one: %s", self.one)
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

        Attributes:
            theta (np.ndarray): Kriging theta values. Shape (k,).
            p (np.ndarray): Kriging p values. Shape (k,).
            Lambda (float): lambda noise value.
            Psi (np.matrix): Correlation matrix Psi. Shape (n,n).
            U (np.matrix): Kriging U matrix, Cholesky decomposition. Shape (n,n).
            negLnLike (float): Negative log likelihood of the surface at the specified hyperparameters.
            pen_val (float): Penalty value.

        Examples:
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                nat_X = np.array([[0], [1]])
                nat_y = np.array([0, 1])
                n=1
                p=1
                S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=False)
                S.initialize_variables(nat_X, nat_y)
                S.set_variable_types()
                print(S.cod_X)
                print(S.cod_y)
                S.set_theta_values()
                print(f"S.theta: {S.theta}")
                S.initialize_matrices()
                S.set_de_bounds()
                new_theta_p_Lambda = S.optimize_model()
                S.extract_from_bounds(new_theta_p_Lambda)
                print(f"S.theta: {S.theta}")
                S.build_Psi()
                print(f"S.Psi: {S.Psi}")
                S.build_U()
                print(f"S.U:{S.U}")
                S.likelihood()
                S.negLnLike

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
            >>> from spotpython.build.kriging import Kriging
                from numpy import power
                import numpy as np
                nat_X = np.array([[0], [1]])
                nat_y = np.array([0, 1])
                n=1
                p=1
                S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=False)
                S.initialize_variables(nat_X, nat_y)
                S.set_variable_types()
                S.set_theta_values()
                print(f"S.theta: {S.theta}")
                print(S.__is_any__(power(10.0, S.theta), 0))
                print(S.__is_any__(S.theta, 0))
                S.theta: [0.]
                False
                True

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

        Attributes:
            Psi (np.matrix): Correlation matrix Psi. Shape (n,n).
            cnd_Psi (float): Condition number of Psi.
            inf_Psi (bool): True if Psi is infinite, False otherwise.

        Examples:
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                nat_X = np.array([[0], [1]])
                nat_y = np.array([0, 1])
                n=1
                p=1
                S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=False)
                S.initialize_variables(nat_X, nat_y)
                S.set_variable_types()
                print(S.nat_X)
                print(S.nat_y)
                S.set_theta_values()
                print(f"S.theta: {S.theta}")
                S.initialize_matrices()
                S.set_de_bounds()
                new_theta_p_Lambda = S.optimize_model()
                S.extract_from_bounds(new_theta_p_Lambda)
                print(f"S.theta: {S.theta}")
                S.build_Psi()
                print(f"S.Psi: {S.Psi}")
                [[0]
                [1]]
                [0 1]
                S.theta: [0.]
                S.theta: [1.72284258]
                S.Psi: [[1.00000001e+00 1.14348852e-23]
                [1.14348852e-23 1.00000001e+00]]

        """
        self.Psi = zeros((self.n, self.n), dtype=float64)
        theta = power(10.0, self.theta)
        if self.n_theta == 1:
            theta = theta * ones(self.k)
        try:
            D = zeros((self.n, self.n))
            if self.ordered_mask.any():
                X_ordered = self.nat_X[:, self.ordered_mask]
                D = squareform(
                    pdist(
                        X_ordered, metric='sqeuclidean', out=None, w=theta[self.ordered_mask]))
            if self.factor_mask.any():
                X_factor = self.nat_X[:, self.factor_mask]
                D = (D + squareform(
                    pdist(X_factor,
                          metric=self.metric_factorial,
                          out=None,
                          w=theta[self.factor_mask])))
            self.Psi = exp(-D)
        except LinAlgError as err:
            print(f"Building Psi failed:\n {self.Psi}. {err=}, {type(err)=}")
        if self.noise:
            logger.debug("In build_Psi(): self.Lambda: %s", self.Lambda)
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

        Attributes:
            U (np.matrix): Kriging U matrix, Cholesky decomposition. Shape (n,n).

        Examples:
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                nat_X = np.array([[0], [1]])
                nat_y = np.array([0, 1])
                n=1
                p=1
                S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=False)
                S.initialize_variables(nat_X, nat_y)
                S.set_variable_types()
                print(S.nat_X)
                print(S.nat_y)
                S.set_theta_values()
                print(f"S.theta: {S.theta}")
                S.initialize_matrices()
                S.set_de_bounds()
                new_theta_p_Lambda = S.optimize_model()
                S.extract_from_bounds(new_theta_p_Lambda)
                print(f"S.theta: {S.theta}")
                S.build_Psi()
                print(f"S.Psi: {S.Psi}")
                S.build_U()
                print(f"S.U:{S.U}")
                [[0]
                [1]]
                [0 1]
                S.theta: [0.]
                S.theta: [1.72284258]
                S.Psi: [[1.00000001e+00 1.14348852e-23]
                [1.14348852e-23 1.00000001e+00]]
                S.U:[[1.00000001e+00 1.14348851e-23]
                [0.00000000e+00 1.00000001e+00]]
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

        Attributes:
            mu (np.float64): Kriging expected mean value mu.
            SigmaSqr (np.float64): Sigma squared value.
            LnDetPsi (np.float64): Determinant Psi matrix.
            negLnLike (float): Negative log likelihood of the surface at the specified hyperparameters.

        Examples:
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                nat_X = np.array([[1], [2]])
                nat_y = np.array([5, 10])
                n=2
                p=1
                S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=False, theta_init_zero=True)
                S.initialize_variables(nat_X, nat_y)
                S.set_variable_types()
                S.set_theta_values()
                S.initialize_matrices()
                S.build_Psi()
                S.build_U()
                S.likelihood()
                # assert S.mu is close to 7.5 with a tolerance of 1e-6
                assert np.allclose(S.mu, 7.5, atol=1e-6)
                E = np.exp(1)
                sigma2 = E/(E**2 -1) * (25/4 + 25/4*E)
                # asssert S.SigmaSqr is close to sigma2 with a tolerance of 1e-6
                assert np.allclose(S.SigmaSqr, sigma2, atol=1e-6)
                print(f"S.LnDetPsi:{S.LnDetPsi}")
                print(f"S.self.negLnLike:{S.negLnLike}")
        """
        # (2.20) in [Forr08a]:
        U_T_inv_one = solve(self.U.T, self.one)
        U_T_inv_cod_y = solve(self.U.T, self.nat_y)
        mu = self.one.T.dot(solve(self.U, U_T_inv_cod_y)) / self.one.T.dot(solve(self.U, U_T_inv_one))
        self.mu = mu
        # (2.31) in [Forr08a]
        cod_y_minus_mu = self.nat_y - self.one.dot(self.mu)
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

        Note:
            * This method provides only a basic plot. For more advanced plots,
                use the `plot_contour()` method of the `Spot` class.

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import analytical
                from spotpython.spot import spot
                # 1-dimensional example
                fun = analytical().fun_sphere
                lower = np.array([-1])
                upper = np.array([1])
                design_control={"init_size": 10}
                S = spot.Spot(fun=fun,
                            noise=False,
                            lower = lower,
                            upper= upper,
                            design_control=design_control,)
                S.initialize_design()
                S.update_stats()
                S.fit_surrogate()
                S.surrogate.plot()
                # 2-dimensional example
                fun = analytical().fun_sphere
                lower = np.array([-1, -1])
                upper = np.array([1, 1])
                design_control={"init_size": 10}
                S = spot.Spot(fun=fun,
                            noise=False,
                            lower = lower,
                            upper= upper,
                            design_control=design_control,)
                S.initialize_design()
                S.update_stats()
                S.fit_surrogate()
                S.surrogate.plot()
        """
        if self.k == 1:
            # TODO: Improve plot (add conf. interval etc.)
            fig = pylab.figure(figsize=(9, 6))
            n_grid = 100
            x = linspace(
                self.min_X[0], self.max_X[0], num=n_grid
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
                self.min_X[0], self.max_X[0], num=n_grid
            )
            y = linspace(
                self.min_X[1], self.max_X[1], num=n_grid
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

            nat_point_X = self.nat_X[:, 0]
            nat_point_Y = self.nat_X[:, 1]
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

    def predict(self, nat_X: ndarray, return_val: str = "y") -> Union[float,
                                                                      Tuple[float, float]]:
        """
        This function returns the prediction (in natural units) of the surrogate at the natural coordinates of X.

        Args:
            self (object):
                The Kriging object.
            nat_X (ndarray):
                Design variable to evaluate in natural units.
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
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                import matplotlib.pyplot as plt
                from numpy import linspace, arange
                rng = np.random.RandomState(1)
                X = linspace(start=0, stop=10, num=1_0).reshape(-1, 1)
                y = np.squeeze(X * np.sin(X))
                training_indices = rng.choice(arange(y.size), size=6, replace=False)
                X_train, y_train = X[training_indices], y[training_indices]
                S = Kriging(name='kriging', seed=124)
                S.fit(X_train, y_train)
                mean_prediction, std_prediction, s_ei = S.predict(X, return_val="all")
                print(f"mean_prediction: {mean_prediction}")
                print(f"std_prediction: {std_prediction}")
                print(f"s_ei: {s_ei}")
                mean_prediction: [-1.41991225e-08  6.48310037e-01  1.76715565e+00 -6.35226564e-01
                                  -4.28585379e+00 -1.22301198e+00  2.49434148e+00  5.61900501e-01
                                  -3.04558205e+00 -5.44021104e+00]
                std_prediction: [3.69706811e-04 2.07958787e+00 3.69706810e-04 3.69706807e-04
                                3.69706809e-04 2.07958584e+00 3.69706811e-04 2.60615408e+00
                                2.60837033e+00 3.69706811e-04]
                s_ei: [-0.00000000e+00 -1.02341235e-03 -0.00000000e+00 -0.00000000e+00
                       -0.00000000e+00 -1.63799181e-02 -0.00000000e+00 -9.45766290e-03
                       -2.53405666e-01 -1.47459347e-04]

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

        Note:
            `self.mu` and `self.SigmaSqr` are computed in `likelihood`, not here.
            See also [Forr08a, p.60].
        """
        self.build_psi_vec(cod_x)
        U_T_inv = solve(self.U.T, self.nat_y - self.one.dot(self.mu))
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

        Attributes:
            self.psi (ndarray):
                psi vector

        Examples:
            >>> import numpy as np
                from spotpython.build.kriging import Kriging
                X_train = np.array([[1., 2.],
                                    [2., 4.],
                                    [3., 6.]])
                y_train = np.array([1., 2., 3.])
                S = Kriging(name='kriging',
                            seed=123,
                            log_level=50,
                            n_theta=1,
                            noise=False,
                            cod_type="norm")
                S.fit(X_train, y_train)
                # force theta to simple values:
                S.theta = np.array([0.0])
                nat_X = np.array([1., 0.])
                S.psi = np.zeros((S.n, 1))
                S.build_psi_vec(nat_X)
                res = np.array([[np.exp(-4)],
                    [np.exp(-17)],
                    [np.exp(-40)]])
                assert np.array_equal(S.psi, res)
                print(f"S.psi: {S.psi}")
                print(f"Control value res: {res}")
                S.psi:
                [[1.83156389e-02]
                [4.13993772e-08]
                [4.24835426e-18]]
                Control value res:
                [[1.83156389e-02]
                [4.13993772e-08]
                [4.24835426e-18]]
        """
        self.psi = zeros((self.n))
        theta = power(10.0, self.theta)
        if self.n_theta == 1:
            theta = theta * ones(self.k)
        try:
            D = zeros((self.n))
            if self.ordered_mask.any():
                X_ordered = self.nat_X[:, self.ordered_mask]
                x_ordered = cod_x[self.ordered_mask]
                D = cdist(x_ordered.reshape(-1, sum(self.ordered_mask)),
                          X_ordered.reshape(-1, sum(self.ordered_mask)),
                          metric='sqeuclidean',
                          out=None,
                          w=theta[self.ordered_mask])
            if self.factor_mask.any():
                X_factor = self.nat_X[:, self.factor_mask]
                x_factor = cod_x[self.factor_mask]
                D = (D + cdist(x_factor.reshape(-1, sum(self.factor_mask)),
                               X_factor.reshape(-1, sum(self.factor_mask)),
                               metric=self.metric_factorial,
                               out=None,
                               w=theta[self.factor_mask]))
            self.psi = exp(-D).T
        except LinAlgError as err:
            print(f"Building psi failed:\n {self.psi}. {err=}, {type(err)=}")

    def weighted_exp_imp(self, cod_x: np.ndarray, w: float) -> float:
        """
        Weighted expected improvement.

        Args:
            self (object): The Kriging object.
            cod_x (np.ndarray): A coded design vector.
            w (float): Weight.

        Returns:
            EI (float): Weighted expected improvement.

        References:
            [Sobester et al. 2005].
        """
        y0, s0 = self.predict_coded(cod_x)
        y_min = min(self.nat_y)
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
