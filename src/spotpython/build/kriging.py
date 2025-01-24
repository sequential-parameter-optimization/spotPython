import copy
from math import erf
import matplotlib.pyplot as plt
from numpy import min, var
from numpy import sqrt
from numpy import exp
from numpy import array
from numpy import log
from numpy import power
from numpy import abs
from numpy import pi
from numpy import spacing
from numpy import append, ndarray, linspace, meshgrid, ravel
from numpy.linalg import cholesky, solve, LinAlgError, cond
from scipy.optimize import differential_evolution
from scipy.linalg import cholesky as scipy_cholesky
import pylab
from spotpython.build.surrogates import surrogates
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from spotpython.utils.repair import repair_non_numeric
from spotpython.utils.aggregate import aggregate_mean_var
import logging
import numpy as np
from typing import List, Union, Tuple, Any, Optional, Dict


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
            theta_init_zero: bool = False,
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
            >>> from spotpython.build import Kriging
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
        self._set_internal_attributes()

        self.noise = noise
        self.var_type = var_type
        self.name = name
        self.seed = seed
        self.log_level = log_level
        self.spot_writer = spot_writer
        self.counter = counter
        self.metric_factorial = metric_factorial
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.min_Lambda = min_Lambda
        self.max_Lambda = max_Lambda
        self.n_theta = n_theta
        self.p_val = p_val
        self.n_p = n_p
        self.optim_p = optim_p
        self.theta_init_zero = theta_init_zero
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

    def _set_internal_attributes(self) -> None:
        """ Set attributes that are not using external arguments that are passed
            to the class constructor.
        """
        self.sigma = 0
        self.eps = sqrt(spacing(1))
        self.min_p = 1
        self.max_p = 2
        # Psi matrix condition:
        self.cnd_Psi = 0
        self.inf_Psi = False

    def _initialize_variables(self, nat_X: np.ndarray, nat_y: np.ndarray) -> None:
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
                nat_X = np.array([[1, 2], [3, 4], [1,2]])
                nat_y = np.array([1, 2, 11])
                S = Kriging()
                S._initialize_variables(nat_X, nat_y)
                print(f"S.nat_X: {S.nat_X}")
                print(f"S.nat_y: {S.nat_y}")
                print(f"S.aggregated_mean_y: {S.aggregated_mean_y}")
                print(f"S.min_X: {S.min_X}")
                print(f"S.max_X: {S.max_X}")
                print(f"S.n: {S.n}")
                print(f"S.k: {S.k}")
                   S.nat_X: [[1 2]
                    [3 4]
                    [1 2]]
                    S.nat_y: [ 1  2 11]
                    S.aggregated_mean_y: [6. 2.]
                    S.min_X: [1 2]
                    S.max_X: [3 4]
                    S.n: 3
                    S.k: 2
        """
        # Validate input dimensions
        if nat_X.ndim != 2 or nat_y.ndim != 1:
            raise ValueError("nat_X must be a 2D array and nat_y must be a 1D array.")
        if nat_X.shape[0] != nat_y.shape[0]:
            raise ValueError("The number of samples in nat_X and nat_y must be equal.")

        # Initialize instance variables
        self.nat_X = copy.deepcopy(nat_X)
        self.nat_y = copy.deepcopy(nat_y)
        self.n, self.k = self.nat_X.shape

        # Calculate and store min and max of X
        self.min_X = np.min(self.nat_X, axis=0)
        self.max_X = np.max(self.nat_X, axis=0)

        # Calculate the aggregated mean of y
        _, aggregated_mean_y, _ = aggregate_mean_var(X=self.nat_X, y=self.nat_y)
        self.aggregated_mean_y = np.copy(aggregated_mean_y)

        # Logging the initialized variables
        logger.debug("In _initialize_variables(): self.nat_X: %s", self.nat_X)
        logger.debug("In _initialize_variables(): self.nat_y: %s", self.nat_y)
        logger.debug("In _initialize_variables(): self.aggregated_mean_y: %s", self.aggregated_mean_y)
        logger.debug("In _initialize_variables(): self.min_X: %s", self.min_X)
        logger.debug("In _initialize_variables(): self.max_X: %s", self.max_X)
        logger.debug("In _initialize_variables(): self.n: %d", self.n)
        logger.debug("In _initialize_variables(): self.k: %d", self.k)

    def _set_variable_types(self) -> None:
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
            >>> from spotpython.build import Kriging
                import numpy as np
                nat_X = np.array([[1, 2], [3, 4], [5, 6]])
                nat_y = np.array([1, 2, 3])
                var_type = ["num", "int", "float"]
                n_theta=2
                n_p=2
                S=Kriging(var_type=var_type, seed=124, n_theta=n_theta, n_p=n_p, optim_p=True, noise=True)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                assert S.var_type == ["num", "int", "float"]
                assert S.num_mask.all() == False
                assert S.factor_mask.all() == False
                assert S.int_mask.all() == False
                assert S.ordered_mask.all() == True
                assert np.all(S.num_mask == np.array([True, False, False]))
                assert np.all(S.int_mask == np.array([False, True, False]))
                assert np.all(S.ordered_mask == np.array([True, True, True]))

        Returns:
            None
        """
        logger.debug("In _set_variable_types(): self.k: %s", self.k)
        logger.debug("In _set_variable_types(): self.var_type: %s", self.var_type)

        # Ensure var_type has appropriate length by defaulting to 'num'
        if len(self.var_type) < self.k:
            self.var_type = ['num'] * self.k  # Corrected to fill with 'num' instead of duplicating
            logger.warning("In _set_variable_types(): All variable types forced to 'num'.")
            logger.debug("In _set_variable_types(): self.var_type: %s", self.var_type)
        # Create masks for each type using numpy vectorized operations
        var_type_array = np.array(self.var_type)
        self.num_mask = (var_type_array == "num")
        self.factor_mask = (var_type_array == "factor")
        self.int_mask = (var_type_array == "int")
        self.ordered_mask = np.isin(var_type_array, ["int", "num", "float"])
        logger.debug("In _set_variable_types(): self.num_mask: %s", self.num_mask)
        logger.debug("In _set_variable_types(): self.factor_mask: %s", self.factor_mask)
        logger.debug("In _set_variable_types(): self.int_mask: %s", self.int_mask)
        logger.debug("In _set_variable_types(): self.ordered_mask: %s", self.ordered_mask)

    def _set_theta_values(self) -> None:
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
            >>> from spotpython.build import Kriging
                import numpy as np
                from numpy import array
                nat_X = np.array([[1, 2], [3, 4]])
                n = nat_X.shape[0]
                k = nat_X.shape[1]
                nat_y = np.array([1, 2])
                n_theta=2
                n_p=2
                S=Kriging(seed=124, n_theta=n_theta, n_p=n_p, optim_p=True, noise=True, theta_init_zero=True)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                S._set_theta_values()
                assert S.theta.all() == array([0., 0.]).all()
                S=Kriging(seed=124, n_theta=n_theta, n_p=n_p, optim_p=True, noise=True, theta_init_zero=False)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                S._set_theta_values()
                t = np.ones(n_theta, dtype=float) * n / (100 * k)
                assert S.theta.all() == t.all()
                nat_X = np.array([[1, 2], [3, 4], [5, 6]])
                n = nat_X.shape[0]
                k = nat_X.shape[1]
                nat_y = np.array([1, 2, 3])
                n_theta=2
                n_p=2
                S=Kriging(seed=124, n_theta=n_theta, n_p=n_p, optim_p=True, noise=True, theta_init_zero=True)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                S._set_theta_values()
                assert S.theta.all() == array([0., 0.]).all()
                S=Kriging(seed=124, n_theta=n_theta, n_p=n_p, optim_p=True, noise=True, theta_init_zero=False)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                S._set_theta_values()
                t = np.ones(n_theta, dtype=float) * n / (100 * k)
                assert S.theta.all() == t.all()
        """
        logger.debug("In set_theta_values(): self.k: %s", self.k)
        logger.debug("In set_theta_values(): self.n_theta: %s", self.n_theta)

        # Adjust `n_theta` if it exceeds `k`
        if self.n_theta > self.k:
            self.n_theta = self.k
            logger.warning("Too few theta values or more theta values than dimensions. `n_theta` set to `k`.")
            logger.debug("In set_theta_values(): self.n_theta reset to: %s", self.n_theta)

        # Initialize theta values
        if hasattr(self, "theta_init_zero") and self.theta_init_zero:
            self.theta = np.zeros(self.n_theta, dtype=float)
            logger.debug("Theta initialized to zeros: %s", self.theta)
        else:
            logger.debug("In set_theta_values(): self.n: %s", self.n)
            self.theta = np.ones(self.n_theta, dtype=float) * self.n / (100 * self.k)
            logger.debug("Theta initialized based on n and k: %s", self.theta)

    def _initialize_matrices(self) -> None:
        """
        Initialize the matrices for the class instance.
        This method initializes several matrices and attributes for the class instance.
        The `p` attribute is initialized as a list of ones with length `n_p`, multiplied by 2.0.
        The `pen_val` attribute is initialized as the natural logarithm of the
        variance of `nat_y`, multiplied by `n`, plus 1e4.
        The `negLnLike`, `LnDetPsi`, `mu`, `U`, `SigmaSqr`, and `Lambda` attributes are all set to None.
        The `Psi` attribute is initialized as a zero matrix with shape `(n, n)` and dtype `float64`.
        The `psi` attribute is initialized as a zero matrix with shape `(n, 1)`.
        The `one` attribute is initialized as a list of ones with length `n`.

        Args:
            self (object): The Kriging object.

        Examples:
            >>> from spotpython.build import Kriging
                import numpy as np
                from numpy import log, var
                nat_X = np.array([[1, 2], [3, 4], [5, 6]])
                nat_y = np.array([1, 2, 3])
                n = nat_X.shape[0]
                k = nat_X.shape[1]
                n_theta=2
                n_p=2
                S=Kriging(seed=124, n_theta=n_theta, n_p=n_p, optim_p=True, noise=True)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                S._set_theta_values()
                S._initialize_matrices()
                assert np.all(S.p == 2.0 * np.ones(n_p))
                # if var(self.nat_y) is > 0, then self.pen_val = self.n * log(var(self.nat_y)) + 1e4
                # else self.pen_val = self.n * var(self.nat_y) + 1e4
                assert S.pen_val == nat_X.shape[0] * log(var(S.nat_y)) + 1e4
                assert S.Psi.shape == (n, n)
                assert S.psi.shape == (n, 1)
                assert S.one.shape == (n,)

        Returns:
            None
        """
        logger.debug("In _initialize_matrices(): self.n_p: %s", self.n_p)

        # Adjust `n_p` if it exceeds `k`
        if self.n_p > self.k:
            self.n_p = self.k
            logger.warning("More p values than dimensions. `n_p` set to `k`.")
            logger.debug("In _initialize_matrices(): self.n_p reset to: %s", self.n_p)

        # Initialize p
        self.p = np.ones(self.n_p) * self.p_val
        logger.debug("In _initialize_matrices(): self.p: %s", self.p)

        # Calculate variance of nat_y
        y_variance = var(self.nat_y)
        logger.debug("In _initialize_matrices(): var(self.nat_y): %s", y_variance)

        # Set penalty value based on variance
        if y_variance > 0:
            self.pen_val = self.n * log(y_variance) + 1e4
        else:
            self.pen_val = self.n * y_variance + 1e4
        logger.debug("In _initialize_matrices(): self.pen_val: %s", self.pen_val)

        # Initialize other attributes
        self.negLnLike = None
        self.LnDetPsi = None
        self.mu = None
        self.U = None
        self.SigmaSqr = None
        self.Lambda = None

        # Initialize matrix Psi and vector psi
        self.Psi = np.zeros((self.n, self.n), dtype=np.float64)
        logger.debug("In _initialize_matrices(): self.Psi shape: %s", self.Psi.shape)

        self.psi = np.zeros((self.n, 1), dtype=np.float64)
        logger.debug("In _initialize_matrices(): self.psi shape: %s", self.psi.shape)

        # Initialize one
        self.one = np.ones(self.n, dtype=np.float64)
        logger.debug("In _initialize_matrices(): self.one: %s", self.one)

    def _set_de_bounds(self) -> None:
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
            >>> from spotpython.build import Kriging
                S = Kriging()
                S._set_de_bounds()
                print(S.de_bounds)
                    [[-3.0, 2.0]]
                S = Kriging(n_theta=2, n_p=2, optim_p=True)
                S._set_de_bounds()
                print(S.de_bounds)
                    [[-3.0, 2.0], [-3.0, 2.0], [1, 2], [1, 2]]
                S = Kriging(n_theta=2, n_p=2, optim_p=True, noise=True)
                S._set_de_bounds()
                print(S.de_bounds)
                    [[-3.0, 2.0], [-3.0, 2.0], [1, 2], [1, 2], [1e-09, 1.0]]
                S = Kriging(n_theta=2, n_p=2, noise=True)
                S._set_de_bounds()
                print(S.de_bounds)
                    [[-3.0, 2.0], [-3.0, 2.0], [1e-09, 1.0]]

        Returns:
            None
        """
        logger.debug("In _set_de_bounds(): self.min_theta: %s", self.min_theta)
        logger.debug("In _set_de_bounds(): self.max_theta: %s", self.max_theta)
        logger.debug("In _set_de_bounds(): self.n_theta: %s", self.n_theta)
        logger.debug("In _set_de_bounds(): self.optim_p: %s", self.optim_p)
        logger.debug("In _set_de_bounds(): self.min_p: %s", self.min_p)
        logger.debug("In _set_de_bounds(): self.max_p: %s", self.max_p)
        logger.debug("In _set_de_bounds(): self.n_p: %s", self.n_p)
        logger.debug("In _set_de_bounds(): self.noise: %s", self.noise)
        logger.debug("In _set_de_bounds(): self.min_Lambda: %s", self.min_Lambda)
        logger.debug("In _set_de_bounds(): self.max_Lambda: %s", self.max_Lambda)

        de_bounds = [[self.min_theta, self.max_theta] for _ in range(self.n_theta)]
        if self.optim_p:
            de_bounds += [[self.min_p, self.max_p] for _ in range(self.n_p)]
            if self.noise:
                de_bounds.append([self.min_Lambda, self.max_Lambda])
        else:
            if self.noise:
                de_bounds.append([self.min_Lambda, self.max_Lambda])
        self.de_bounds = de_bounds
        logger.debug("In _set_de_bounds(): self.de_bounds: %s", self.de_bounds)

    def _optimize_model(self) -> Union[List[float], Tuple[float]]:
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
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                S._set_theta_values()
                S._initialize_matrices()
                S._set_de_bounds()
                new_theta_p_Lambda = S._optimize_model()
                print(new_theta_p_Lambda)
                    [0.12167915 1.49467909 1.82808259 1.69648798 0.79564346]
            >>> from spotpython.build import Kriging
                import numpy as np
                nat_X = np.array([[1, 2], [3, 4]])
                nat_y = np.array([1, 2])
                n_theta=2
                n_p=2
                S=Kriging(seed=124, n_theta=n_theta, n_p=n_p, optim_p=True, noise=True)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                S._set_theta_values()
                S._initialize_matrices()
                S._set_de_bounds()
                new_theta_p_Lambda = S._optimize_model()
                assert  len(new_theta_p_Lambda) == n_theta + n_p + 1
            >>> from spotpython.build import Kriging
                import numpy as np
                nat_X = np.array([[1, 2], [3, 4]])
                nat_y = np.array([1, 2])
                n_theta=2
                n_p=2
                S=Kriging(seed=124, n_theta=n_theta, n_p=n_p, optim_p=True, noise=False)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                S._set_theta_values()
                S._initialize_matrices()
                S._set_de_bounds()
                new_theta_p_Lambda = S._optimize_model()
                assert len(new_theta_p_Lambda) == n_theta + n_p
            >>> from spotpython.build import Kriging
                import numpy as np
                nat_X = np.array([[1, 2], [3, 4]])
                nat_y = np.array([1, 2])
                n_theta=2
                n_p=1
                S=Kriging(seed=124, n_theta=n_theta, n_p=n_p, optim_p=True, noise=False)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                S._set_theta_values()
                S._initialize_matrices()
                S._set_de_bounds()
                new_theta_p_Lambda = S._optimize_model()
                assert  len(new_theta_p_Lambda) == n_theta + n_p
            >>> from spotpython.build import Kriging
                import numpy as np
                nat_X = np.array([[1, 2], [3, 4]])
                nat_y = np.array([1, 2])
                n_theta=1
                n_p=1
                S=Kriging(seed=124, n_theta=n_theta, n_p=n_p, optim_p=False, noise=False)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                S._set_theta_values()
                S._initialize_matrices()
                S._set_de_bounds()
                new_theta_p_Lambda = S._optimize_model()
                assert  len(new_theta_p_Lambda) == 1

        Returns:
            result["x"] (Union[List[float], Tuple[float]]):
                A list or tuple of optimized parameter values.
        """
        logger.debug("Entering _optimize_model.")
        if not callable(self.model_optimizer):
            logger.error("model_optimizer is not callable.")
            raise ValueError("model_optimizer must be a callable function or method.")

        optimizer_strategies: Dict[str, Dict] = {
            'dual_annealing': {'func': self.fun_likelihood, 'bounds': self.de_bounds},
            'differential_evolution': {
                'func': self.fun_likelihood,
                'bounds': self.de_bounds,
                'maxiter': self.model_fun_evals,
                'seed': self.seed
            },
            'direct': {
                'func': self.fun_likelihood,
                'bounds': self.de_bounds,
                'eps': 1e-2
            },
            'shgo': {'func': self.fun_likelihood, 'bounds': self.de_bounds},
            'basinhopping': {'func': self.fun_likelihood, 'x0': np.mean(self.de_bounds, axis=1)}
        }

        optimizer_name = self.model_optimizer.__name__
        logger.debug("Optimizer selected: %s", optimizer_name)

        if optimizer_name not in optimizer_strategies:
            logger.info("Using default options for optimizer: %s", optimizer_name)
            optimizer_args = {'func': self.fun_likelihood, 'bounds': self.de_bounds}
        else:
            optimizer_args = optimizer_strategies[optimizer_name]

        logger.debug("Parameters for optimization: %s", optimizer_args)

        try:
            result = self.model_optimizer(**optimizer_args)
        except Exception as e:
            logger.error("Optimization failed due to error: %s", str(e))
            raise

        if "x" not in result:
            logger.error("Optimization result does not contain 'x'. Result: %s", result)
            raise ValueError("The optimization result does not contain the expected 'x' key.")
        logger.debug("Optimization result: %s", result)
        optimized_parameters = list(result["x"])
        logger.debug("Extracted optimized parameters: %s", optimized_parameters)
        return optimized_parameters

    def _extract_from_bounds(self, new_theta_p_Lambda: np.ndarray) -> None:
        """
        Extract `theta`, `p`, and `Lambda` from bounds. The kriging object stores
        `theta` as an array,  `p` as an array, and `Lambda` as a float.

        Args:
            self (object): The Kriging object.
            new_theta_p_Lambda (np.ndarray):
                1d-array with theta, p, and Lambda values. Order is important.
        Returns:
            None

        Examples:
            >>> import numpy as np
                from spotpython.build import Kriging
                num_theta = 2
                num_p = 3
                S = Kriging(
                    seed=124,
                    n_theta=num_theta,
                    n_p=num_p,
                    optim_p=True,
                    noise=True
                )
                bounds_array = np.array([1, 2, 3, 4, 5, 6])
                S._extract_from_bounds(new_theta_p_Lambda=bounds_array)
                assert np.array_equal(S.theta,
                    [1, 2]), f"Expected theta to be [1, 2] but got {S.theta}"
                assert np.array_equal(S.p,
                    [3, 4, 5]), f"Expected p to be [3, 4, 5] but got {S.p}"
                assert S.Lambda == 6, f"Expected Lambda to be 6 but got {S.Lambda}"
            >>> import numpy as np
                from spotpython.build import Kriging
                num_theta = 1
                num_p = 1
                S = Kriging(
                    seed=124,
                    n_theta=num_theta,
                    n_p=num_p,
                    optim_p=False,
                    noise=False
                )
                bounds_array = np.array([1])
                S._extract_from_bounds(new_theta_p_Lambda=bounds_array)
                assert np.array_equal(S.theta,
                    [1]), f"Expected theta to be [1] but got {S.theta}"
            >>> import numpy as np
                from spotpython.build import Kriging
                num_theta = 1
                num_p = 2
                S = Kriging(
                    seed=124,
                    n_theta=num_theta,
                    n_p=num_p,
                    optim_p=True,
                    noise=True
                )
                bounds_array = np.array([1, 2, 3, 4])
                S._extract_from_bounds(new_theta_p_Lambda=bounds_array)
                assert np.array_equal(S.theta,
                    [1]), f"Expected theta to be [1, 2] but got {S.theta}"
                assert np.array_equal(S.p,
                    [2, 3]), f"Expected p to be [3, 4, 5] but got {S.p}"
                assert S.Lambda == 4, f"Expected Lambda to be 6 but got {S.Lambda}"

        """
        logger.debug("Extracting parameters from: %s", new_theta_p_Lambda)

        # Validate array length
        expected_length = self.n_theta
        if self.optim_p:
            expected_length += self.n_p
        if self.noise:
            expected_length += 1

        if len(new_theta_p_Lambda) < expected_length:
            logger.error("Input array is too short. Expected at least %d elements, got %d.",
                         expected_length, len(new_theta_p_Lambda))
            raise ValueError(f"Input array must have at least {expected_length} elements.")

        # Extract theta
        self.theta = new_theta_p_Lambda[:self.n_theta]
        logger.debug("Extracted theta: %s", self.theta)

        if self.optim_p:
            # Extract p if optim_p is True
            self.p = new_theta_p_Lambda[self.n_theta:self.n_theta + self.n_p]
            logger.debug("Extracted p: %s", self.p)

        if self.noise:
            # Extract Lambda
            lambda_index = self.n_theta + (self.n_p if self.optim_p else 0)
            self.Lambda = new_theta_p_Lambda[lambda_index]
            logger.debug("Extracted Lambda: %s", self.Lambda)

    def build_Psi(self) -> None:
        """
        Constructs a new (n x n) correlation matrix Psi to reflect new data
        or a change in hyperparameters.
        This method uses `theta`, `p`, and coded `X` values to construct the
        correlation matrix as described in [Forr08a, p.57].

        Attributes:
            Psi (np.matrix): Correlation matrix Psi. Shape (n,n).
            cnd_Psi (float): Condition number of Psi.
            inf_Psi (bool): True if Psi is infinite, False otherwise.

        Raises:
            LinAlgError: If building Psi fails.

        Examples:
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                nat_X = np.array([[0], [1]])
                nat_y = np.array([0, 1])
                n=1
                p=1
                S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=False)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                print(S.nat_X)
                print(S.nat_y)
                S._set_theta_values()
                print(f"S.theta: {S.theta}")
                S._initialize_matrices()
                S._set_de_bounds()
                new_theta_p_Lambda = S._optimize_model()
                S._extract_from_bounds(new_theta_p_Lambda)
                print(f"S.theta: {S.theta}")
                S.build_Psi()
                print(f"S.Psi: {S.Psi}")
                    [[0]
                    [1]]
                    [0 1]
                    S.theta: [0.]
                    S.theta: [1.60036366]
                    S.Psi: [[1.00000001e+00 4.96525625e-18]
                    [4.96525625e-18 1.00000001e+00]]
        """
        try:
            n = self.n
            k = self.k
            theta10 = np.power(10.0, self.theta)

            # Ensure theta has the correct length
            if self.n_theta == 1:
                theta10 = theta10 * np.ones(k)

            # Initialize the Psi matrix
            self.Psi = np.zeros((n, n), dtype=np.float64)

            # Calculate the distance matrix using ordered variables
            if self.ordered_mask.any():
                X_ordered = self.nat_X[:, self.ordered_mask]
                D_ordered = squareform(
                    pdist(X_ordered, metric='sqeuclidean', w=theta10[self.ordered_mask])
                )
                self.Psi += D_ordered

            # Add the contribution of factor variables to the distance matrix
            if self.factor_mask.any():
                X_factor = self.nat_X[:, self.factor_mask]
                D_factor = squareform(
                    pdist(X_factor, metric=self.metric_factorial, w=theta10[self.factor_mask])
                )
                self.Psi += D_factor

            # Calculate correlation from distance
            self.Psi = np.exp(-self.Psi)

            # Adjust diagonal elements for noise or minimum epsilon
            diag_indices = np.diag_indices_from(self.Psi)
            if self.noise:
                self.Psi[diag_indices] += self.Lambda
                logger.debug("Noise level Lambda applied to diagonal: %s", self.Lambda)
            else:
                self.Psi[diag_indices] += self.eps

            # Check for infinite values
            self.inf_Psi = np.isinf(self.Psi).any()

            # Calculate condition number
            self.cnd_Psi = cond(self.Psi)
            logger.debug("Condition number of Psi: %f", self.cnd_Psi)

        except LinAlgError as err:
            logger.error("Building Psi failed. Error: %s, Type: %s", err, type(err))
            raise

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
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                print(S.nat_X)
                print(S.nat_y)
                S._set_theta_values()
                print(f"S.theta: {S.theta}")
                S._initialize_matrices()
                S._set_de_bounds()
                new_theta_p_Lambda = S._optimize_model()
                S._extract_from_bounds(new_theta_p_Lambda)
                print(f"S.theta: {S.theta}")
                S.build_Psi()
                print(f"S.Psi: {S.Psi}")
                S.build_U()
                print(f"S.U:{S.U}")
                    [[0]
                    [1]]
                    [0 1]
                    S.theta: [0.]
                    S.theta: [1.60036366]
                    S.Psi: [[1.00000001e+00 4.96525625e-18]
                    [4.96525625e-18 1.00000001e+00]]
                    S.U:[[1.00000001e+00 4.96525622e-18]
                    [0.00000000e+00 1.00000001e+00]]
        """
        try:
            self.U = scipy_cholesky(self.Psi, lower=True) if scipy else cholesky(self.Psi)
            self.U = self.U.T
        except LinAlgError as err:
            print(f"build_U() Cholesky failed for Psi:\n {self.Psi}. {err=}, {type(err)=}")

    def fit(self, nat_X: np.ndarray, nat_y: np.ndarray) -> object:
        """
        Fits the hyperparameters (`theta`, `p`, `Lambda`) of the Kriging model.
        The function computes the following internal values:
        1. `theta`, `p`, and `Lambda` values via optimization of the function `fun_likelihood()`.
        2. Correlation matrix `Psi` via `buildPsi()`.
        3. U matrix via `buildU()`.

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
            >>> from spotpython.build import Kriging
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
        self._initialize_variables(nat_X, nat_y)
        self._set_variable_types()
        self._set_theta_values()
        self._initialize_matrices()
        # build_Psi() and build_U() are called in fun_likelihood
        self._set_de_bounds()
        # Finally, set new theta and p values and update the surrogate again
        # for new_theta_p_Lambda in de_results["x"]:
        new_theta_p_Lambda = self._optimize_model()
        self._extract_from_bounds(new_theta_p_Lambda)
        self.build_Psi()
        self.build_U()
        # TODO: check if the following line is necessary!
        self.likelihood()
        self.update_log()

    def predict(self, nat_X: ndarray, return_val: str = "y") -> Union[float, Tuple[float, float]]:
        """
        This function returns the prediction (in natural units) of the surrogate at the natural coordinates of X.

        Args:
            self (object): The Kriging object.
            nat_X (ndarray): Design variable to evaluate in natural units.
            return_val (str): Specifies which prediction values to return. It can be "y", "s", "ei", or "all".

        Returns:
            Union[float, Tuple[float, float, float]]: Depending on `return_val`, returns the predicted value,
            predicted error, expected improvement, or all.

        Raises:
            TypeError: If `nat_X` is not an ndarray or doesn't match expected dimensions.

        Examples:
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
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
        """
        if not isinstance(nat_X, ndarray):
            raise TypeError(f"Expected an ndarray, got {type(nat_X)} instead.")

        try:
            X = nat_X.reshape(-1, self.nat_X.shape[1])
            X = repair_non_numeric(X, self.var_type)
        except Exception as e:
            raise TypeError("Input to predict was not convertible to the size of X") from e

        y, s, ei = self.predict_coded_batch(X)

        if return_val == "y":
            return y
        elif return_val == "s":
            return s
        elif return_val == "ei":
            return -ei
        elif return_val == "all":
            return y, s, -ei
        else:
            raise ValueError(f"Invalid return_val: {return_val}. Supported values are 'y', 's', 'ei', 'all'.")

    def predict_coded(self, cod_x: np.ndarray) -> Tuple[float, float, float]:
        """
        Kriging prediction of one point in coded units as described in (2.20) in [Forr08a].
        The error is returned as well. The method is used in `predict`.

        Args:
            self (object): The Kriging object.
            cod_x (np.ndarray): Point in coded units to make prediction at.

        Returns:
            Tuple[float, float, float]: Predicted value, predicted error, and expected improvement.

        Note:
            Uses attributes such as `self.mu` and `self.SigmaSqr` that are expected
            to be calculated by `likelihood`.

        Examples:
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                from numpy import linspace, arange, empty
                rng = np.random.RandomState(1)
                X = linspace(start=0, stop=10, num=10).reshape(-1, 1)
                y = np.squeeze(X * np.sin(X))
                training_indices = rng.choice(arange(y.size), size=6, replace=False)
                X_train, y_train = X[training_indices], y[training_indices]
                S = Kriging(name='kriging', seed=124)
                S.fit(X_train, y_train)
                n = X.shape[0]
                y = empty(n, dtype=float)
                s = empty(n, dtype=float)
                ei = empty(n, dtype=float)
                for i in range(n):
                    y_coded, s_coded, ei_coded = S.predict_coded(X[i, :])
                    y[i] = y_coded if np.isscalar(y_coded) else y_coded.item()
                    s[i] = s_coded if np.isscalar(s_coded) else s_coded.item()
                    ei[i] = ei_coded if np.isscalar(ei_coded) else ei_coded.item()
                print(f"y: {y}")
                print(f"s: {s}")
                print(f"ei: {-1.0*ei}")
        """
        self.build_psi_vec(cod_x)
        mu_adj = self.mu
        psi = self.psi

        # Calculate the prediction
        U_T_inv = solve(self.U.T, self.nat_y - self.one.dot(mu_adj))
        f = mu_adj + psi.T.dot(solve(self.U, U_T_inv))[0]

        Lambda = self.Lambda if self.noise else 0.0

        # Calculate the estimated error
        SSqr = self.SigmaSqr * (1 + Lambda - psi.T.dot(solve(self.U, solve(self.U.T, psi))))
        SSqr = power(abs(SSqr), 0.5)[0]

        # Calculate expected improvement
        EI = self.exp_imp(y0=f, s0=SSqr)

        return f, SSqr, EI

    def predict_coded_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized prediction for batch input using coded units.

        Args:
            X (np.ndarray): Input array of coded points.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                Arrays of predicted values, predicted errors, and expected improvements.
        """
        n = X.shape[0]
        y = np.empty(n, dtype=float)
        s = np.empty(n, dtype=float)
        ei = np.empty(n, dtype=float)

        for i in range(n):
            y_coded, s_coded, ei_coded = self.predict_coded(X[i, :])
            y[i] = y_coded if np.isscalar(y_coded) else y_coded.item()
            s[i] = s_coded if np.isscalar(s_coded) else s_coded.item()
            ei[i] = ei_coded if np.isscalar(ei_coded) else ei_coded.item()

        return y, s, ei

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
        # We do not use the min y values, but the aggregated mean values
        # y_min = min(self.nat_y)
        y_min = min(self.aggregated_mean_y)
        if s0 <= 0.0:
            EI = 0.0
        elif s0 > 0.0:
            # Ensure (y_min - y0) / s0 is a scalar
            diff_scaled = (y_min - y0) / s0
            # Calculate expected improvement components
            EI_one = (y_min - y0) * (0.5 + 0.5 * erf((1.0 / sqrt(2.0)) * diff_scaled))
            EI_two = (s0 * (1.0 / sqrt(2.0 * pi))) * exp(-(1.0 / 2.0) * diff_scaled ** 2)

            EI = EI_one + EI_two

        return EI

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
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                S._set_theta_values()
                S._initialize_matrices()
                S._set_de_bounds()
                new_theta_p_Lambda = S._optimize_model()
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
                self.spot_writer.add_scalars("spot_p",
                                             {f"p_{i}": p[i] for i in range(self.n_p)}, self.counter+self.log_length)
            self.spot_writer.flush()

    def fun_likelihood(self, new_theta_p_Lambda: np.ndarray) -> float:
        """
        Compute log likelihood for a set of hyperparameters (theta, p, Lambda).

        This method computes the log likelihood for a set of hyperparameters
        (theta, p, Lambda) using several internal methods for matrix construction
        and likelihood evaluation. It handles potential errors by returning a
        penalty value for non-computable states.

        Args:
            new_theta_p_Lambda (np.ndarray): An array containing `theta`, `p`, and `Lambda` values.

        Returns:
            float: The negative log likelihood or the penalty value if computation fails.

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
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                print(S.nat_X)
                print(S.nat_y)
                S._set_theta_values()
                print(f"S.theta: {S.theta}")
                S._initialize_matrices()
                S._set_de_bounds()
                new_theta_p_Lambda = S._optimize_model()
                S._extract_from_bounds(new_theta_p_Lambda)
                print(f"S.theta: {S.theta}")
                S.build_Psi()
                print(f"S.Psi: {S.Psi}")
                S.build_U()
                print(f"S.U:{S.U}")
                S.likelihood()
                S.negLnLike
                    [[0]
                    [1]]
                    [0 1]
                    S.theta: [0.]
                    S.theta: [1.60036366]
                    S.Psi: [[1.00000001e+00 4.96525625e-18]
                    [4.96525625e-18 1.00000001e+00]]
                    S.U:[[1.00000001e+00 4.96525622e-18]
                    [0.00000000e+00 1.00000001e+00]]
                    -1.3862943611198906
        """
        # Extract hyperparameters
        self._extract_from_bounds(new_theta_p_Lambda)
        # Check transformed theta values
        theta10 = np.power(10.0, self.theta)
        if self.__is_any__(theta10, 0):
            logger.warning("Failure in fun_likelihood: 10^theta == 0. Setting negLnLike to %s", self.pen_val)
            return self.pen_val
        # Build Psi matrix and check its condition
        self.build_Psi()
        if getattr(self, 'inf_Psi', False) or getattr(self, 'cnd_Psi', float('inf')) > 1e9:
            logger.warning("Failure in fun_likelihood: Psi is ill-conditioned: %s", getattr(self, 'cnd_Psi', 'unknown'))
            logger.warning("Setting negLnLike to: %s", self.pen_val)
            return self.pen_val
        # Build U matrix and handle exceptions
        try:
            self.build_U()
        except Exception as error:
            logger.error("Error in fun_likelihood(). Call to build_U() failed: %s", error)
            logger.error("Setting negLnLike to %.2f.", self.pen_val)
            return self.pen_val

        # Calculate likelihood
        self.likelihood()
        return self.negLnLike

    def __is_any__(self, x: Union[np.ndarray, Any], v: Any) -> bool:
        """
        Check if any element in `x` is equal to `v`.

        This method checks if any element in the input array-like `x`
        is equal to the given value `v`. Converts inputs to numpy arrays as necessary.

        Args:
            x (Union[np.ndarray, Any]): The input array-like object to check.
            v (Any): The value to check for in `x`.

        Returns:
            bool: True if any element in `x` is equal to `v`, False otherwise.

        Examples:
            >>> from spotpython.build.kriging import Kriging
                from numpy import power
                import numpy as np
                nat_X = np.array([[0], [1]])
                nat_y = np.array([0, 1])
                n=1
                p=1
                S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=False)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                S._set_theta_values()
                print(f"S.theta: {S.theta}")
                print(S.__is_any__(power(10.0, S.theta), 0))
                print(S.__is_any__(S.theta, 0))
                    S.theta: [0.]
                    False
                    True
        """

        if not isinstance(x, np.ndarray):
            x = np.array([x])  # Wrap scalar x in an array
        return np.any(x == v)

    def likelihood(self) -> None:
        """
        Calculate the negative concentrated log-likelihood.
        Implements equation (2.32) from [Forr08a] to compute the negative of the
        concentrated log-likelihood. Updates `mu`, `SigmaSqr`, `LnDetPsi`, and `negLnLike`.

        Note:
            Requires prior calls to `build_Psi` and `build_U`.

        Attributes:
            mu (np.float64): Kriging expected mean value mu.
            SigmaSqr (np.float64): Sigma squared value.
            LnDetPsi (np.float64): Logarithm of the determinant of Psi matrix.
            negLnLike (float): Negative log likelihood of the surface at the specified hyperparameters.

        Raises:
            LinAlgError: If matrix operations fail.

        Examples:
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                nat_X = np.array([[1], [2]])
                nat_y = np.array([5, 10])
                n=2
                p=1
                S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=False, theta_init_zero=True)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                S._set_theta_values()
                S._initialize_matrices()
                S.build_Psi()
                S.build_U()
                S.likelihood()
                assert np.allclose(S.mu, 7.5, atol=1e-6)
                E = np.exp(1)
                sigma2 = E / (E**2 - 1) * (25/4 + 25/4*E)
                assert np.allclose(S.SigmaSqr, sigma2, atol=1e-6)
                print(f"S.LnDetPsi:{S.LnDetPsi}")
                print(f"S.negLnLike:{S.negLnLike}")
                    S.LnDetPsi:-0.1454134234019476
                    S.negLnLike:2.2185498738611282
        """
        try:
            # Solving linear equations for needed components
            U_T_inv_one = solve(self.U.T, self.one)
            U_T_inv_nat_y = solve(self.U.T, self.nat_y)
            # Mean calculation: (2.20) in [Forr08a]
            self.mu = (self.one.T @ solve(self.U, U_T_inv_nat_y)) / (self.one.T @ solve(self.U, U_T_inv_one))
            # Residuals
            cod_y_minus_mu = self.nat_y - self.one * self.mu
            # Sigma squared calculation: (2.31) in [Forr08a]
            self.SigmaSqr = (cod_y_minus_mu.T @ solve(self.U, solve(self.U.T, cod_y_minus_mu))) / self.n
            # Log determinant of Psi: (2.32) in [Forr08a]
            self.LnDetPsi = 2.0 * np.sum(np.log(np.abs(np.diag(self.U))))
            # Negative log-likelihood calculation: simplified from (2.32)
            self.negLnLike = 0.5 * (self.n * np.log(self.SigmaSqr) + self.LnDetPsi)
            logger.debug("Likelihood calculated: mu=%s, SigmaSqr=%s, LnDetPsi=%s, negLnLike=%s",
                         self.mu, self.SigmaSqr, self.LnDetPsi, self.negLnLike)
        except LinAlgError as error:
            logger.error("LinAlgError in likelihood calculation: %s", error)
            raise

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
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import spot
                from spotpython.utils.init import fun_control_init, design_control_init
                # 1-dimensional example
                fun = analytical().fun_sphere
                fun_control=fun_control_init(lower = np.array([-1]),
                                            upper = np.array([1]),
                                            noise=False)
                design_control=design_control_init(init_size=10)
                S = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control)
                S.initialize_design()
                S.update_stats()
                S.fit_surrogate()
                S.surrogate.plot()
                # 2-dimensional example
                fun = analytical().fun_sphere
                fun_control=fun_control_init(lower = np.array([-1, -1]),
                                            upper = np.array([1, 1]),
                                            noise=False)
                design_control=design_control_init(init_size=10)
                S = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control)
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

    def build_psi_vec(self, cod_x: np.ndarray) -> None:
        """
        Build the psi vector required for predictive methods.

        Args:
            cod_x (ndarray): Point to calculate the psi vector for.

        Returns:
            None

        Modifies:
            self.psi (np.ndarray): Updates the psi vector.

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
        """
        logger.debug("Building psi vector for point: %s", cod_x)
        try:
            self.psi = np.zeros((self.n, 1))
            theta10 = np.power(10.0, self.theta)
            if self.n_theta == 1:
                theta10 = theta10 * np.ones(self.k)

            D = np.zeros(self.n)

            # Compute ordered distance contributions
            if self.ordered_mask.any():
                X_ordered = self.nat_X[:, self.ordered_mask]
                x_ordered = cod_x[self.ordered_mask]
                D += cdist(x_ordered.reshape(1, -1),
                           X_ordered,
                           metric='sqeuclidean',
                           w=theta10[self.ordered_mask]).ravel()
            logger.debug("Distance D after ordered mask: %s", D)
            # Compute factor distance contributions
            if self.factor_mask.any():
                X_factor = self.nat_X[:, self.factor_mask]
                x_factor = cod_x[self.factor_mask]
                D += cdist(x_factor.reshape(1, -1),
                           X_factor,
                           metric=self.metric_factorial,
                           w=theta10[self.factor_mask]).ravel()
            logger.debug("Distance D after factor mask: %s", D)

            self.psi = np.exp(-D).reshape(-1, 1)

        except np.linalg.LinAlgError as err:
            logger.error("Building psi failed due to a linear algebra error: %s. Error type: %s", err, type(err))

    def weighted_exp_imp(self, cod_x: np.ndarray, w: float) -> float:
        """
        Weighted expected improvement. Currently not used in `spotpython`

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
