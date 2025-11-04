import numpy as np
from numpy.linalg import LinAlgError, cond
from typing import Dict, Tuple, List, Optional, Callable, Union
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.special import erf
import matplotlib.pyplot as plt
from numpy import linspace, append
from scipy.spatial.distance import cdist, pdist, squareform
from spotpython.surrogate.plot import plotkd

# --- Kernel functions ---


def gauss_kernel(D):
    """Gaussian (RBF) kernel: exp(-D)"""
    return np.exp(-D)


def matern_kernel(D, nu=2.5):
    """Matern kernel (default nu=2.5, smooth)."""
    if nu == 0.5:
        return np.exp(-np.sqrt(D))
    elif nu == 1.5:
        sqrt3D = np.sqrt(3.0 * D)
        return (1.0 + sqrt3D) * np.exp(-sqrt3D)
    elif nu == 2.5:
        sqrt5D = np.sqrt(5.0 * D)
        return (1.0 + sqrt5D + (5.0 / 3.0) * D) * np.exp(-sqrt5D)
    else:
        # Fallback to Gaussian for unsupported nu
        return np.exp(-D)


def exponential_kernel(D):
    """Exponential kernel: exp(-sqrt(D))"""
    return np.exp(-np.sqrt(D))


def cubic_kernel(D):
    """Cubic kernel: 1 - D^3"""
    return 1.0 - D**3


def linear_kernel(D):
    """Linear kernel: 1 - D"""
    return 1.0 - D


def rational_quadratic_kernel(D, alpha=1.0):
    """Rational Quadratic kernel: (1 + D/(2*alpha))^(-alpha)"""
    return (1.0 + D / (2.0 * alpha)) ** (-alpha)


def poly_kernel(D, degree=2):
    """Polynomial kernel: (1 + D)^degree"""
    return (1.0 + D) ** degree


# --- The New Kriging Class with Nyström Approximation as introduced in v0.34.0 ---


class Kriging(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible Kriging model class for regression tasks,
    extended with an optional Nyström approximation for scaling and explicit kernel selection.

    Public API and core behavior match src/spotpython/surrogate/kriging.py.
    The same basis function/correlation definitions are used. When use_nystrom=True,
    training and prediction use low-rank Woodbury solves based on m inducing points.

    Kernel selection is explicit via the `kernel` and `kernel_params` arguments.

    Attributes:
        eps (float): A small regularization term to reduce ill-conditioning.
        penalty (float): The penalty value used if the correlation matrix is ill-conditioned.
        logtheta_loglambda_p_ (np.ndarray): Best-fit log(theta), log(lambda), and p parameters from fit().
        U_ (np.ndarray): Cholesky factor of the correlation matrix (exact mode).
        X_ (np.ndarray): Training input data (n x k).
        y_ (np.ndarray): Training target values (n,).
        negLnLike (float): Negative log-likelihood at the optimum.
        Psi_ (np.ndarray): Correlation matrix used in likelihood (exact mode).
        method (str): "interpolation", "regression", or "reinterpolation".
        isotropic (bool): If True, one theta for all ordered dims.
        use_nystrom (bool): If True, use Nyström low-rank solves.
        nystrom_m (int): Number of inducing points (landmarks).
        nystrom_seed (int): RNG seed for landmark selection.
        kernel (str or callable): Kernel type ("gauss", "matern", "exp", "cubic", "linear", "rq", "poly") or a custom callable.
        kernel_params (dict): Parameters for the kernel (e.g., nu for Matern, alpha for rational quadratic, degree for poly).

    Notes (Forrester et al., Engineering Design via Surrogate Modelling, Ch. 2/3/6):
        - Correlation: R = kernel(D), with D weighted distances (Ch. 2).
        - Ordinary Kriging μ, σ^2, concentrated likelihood (Ch. 3).
        - Prediction f̂, variance s^2, and Expected Improvement (Ch. 3 & 6).

    Examples:
        >>> import numpy as np
        >>> from spotpython.surrogate.kriging import Kriging
        >>> X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        >>> y_train = np.array([0.1, 0.2, 0.3])
        >>> # Gaussian kernel (default)
        >>> model = Kriging(kernel="gauss").fit(X_train, y_train)
        >>> # Matern kernel (nu=1.5)
        >>> model_matern = Kriging(kernel="matern", kernel_params={"nu": 1.5}).fit(X_train, y_train)
        >>> # Rational Quadratic kernel
        >>> model_rq = Kriging(kernel="rq", kernel_params={"alpha": 2.0}).fit(X_train, y_train)
        >>> # Custom kernel
        >>> def custom_kernel(D): return np.exp(-D**2)
        >>> model_custom = Kriging(kernel=custom_kernel).fit(X_train, y_train)
    """

    def __init__(
        self,
        eps: float = None,
        penalty: float = 1e4,
        method: str = "regression",
        var_type: List[str] = ["num"],
        name: str = "Kriging",
        seed: int = 124,
        model_optimizer=None,
        model_fun_evals: Optional[int] = None,
        n_theta: Optional[int] = None,
        min_theta: float = -3.0,
        max_theta: float = 2.0,
        theta_init_zero: bool = False,
        p_val: float = 2.0,
        n_p: int = 1,
        optim_p: bool = False,
        min_p: float = 1.0,
        max_p: float = 2.0,
        min_Lambda: float = -9.0,
        max_Lambda: float = 0.0,
        log_level: int = 50,
        spot_writer=None,
        counter=None,
        metric_factorial: str = "canberra",
        isotropic: bool = False,
        theta: Optional[np.ndarray] = None,
        Lambda: Optional[float] = None,
        # Nyström options
        use_nystrom: bool = False,
        nystrom_m: Optional[int] = None,
        nystrom_seed: int = 1234,
        # Kernel options
        kernel: Union[str, Callable] = "gauss",
        kernel_params: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the Kriging model.

        Args:
            eps (float): Small regularization term for numerical stability.
            penalty (float): Penalty value for ill-conditioned correlation matrices.
            method (str): "interpolation", "regression", or "reinterpolation".
            var_type (List[str]): Variable types for each dimension.
            name (str): Name of the model.
            seed (int): Random seed for reproducibility.
            model_optimizer: Optimizer function for hyperparameter tuning.
            model_fun_evals (Optional[int]): Max function evaluations for optimizer.
            n_theta (Optional[int]): Number of theta parameters.
            min_theta (float): Minimum log10(theta) bound.
            max_theta (float): Maximum log10(theta) bound.
            theta_init_zero (bool): If True, initialize theta at zero.
            p_val (float): Initial p value for correlation function.
            n_p (int): Number of p parameters if optim_p is True.
            optim_p (bool): If True, optimize p parameters.
            min_p (float): Minimum p bound.
            max_p (float): Maximum p bound.
            min_Lambda (float): Minimum log10(Lambda) bound.
            max_Lambda (float): Maximum log10(Lambda) bound.
            log_level (int): Logging level.
            spot_writer: Writer object for logging metrics.
            counter: Counter for logging steps.
            metric_factorial (str): Distance metric for factor variables.
            isotropic (bool): If True, use isotropic theta.
            theta (Optional[np.ndarray]): Initial theta values.
            Lambda (Optional[float]): Initial Lambda value.
            use_nystrom (bool): If True, use Nyström approximation.
            nystrom_m (Optional[int]): Number of landmarks for Nyström.
            nystrom_seed (int): Seed for landmark selection in Nyström.
            kernel (str or callable): Kernel type ("gauss", "matern", "exp", "cubic", "linear", "rq", "poly") or a custom callable.
            kernel_params (dict): Parameters for the kernel (e.g., nu for Matern, alpha for rational quadratic, degree for poly).
        """
        if eps is None:
            self.eps = self._get_eps()
        else:
            if eps <= 0:
                raise ValueError("eps must be positive")
            self.eps = eps
        self.penalty = penalty
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
        self.min_p = min_p
        self.max_p = max_p
        self.n_theta = n_theta  # Will be set during fit
        self.isotropic = isotropic
        self.p_val = p_val
        self.n_p = n_p
        self.optim_p = optim_p
        self.theta_init_zero = theta_init_zero
        self.theta = theta
        self.Lambda = Lambda
        self.model_optimizer = model_optimizer or differential_evolution
        self.model_fun_evals = model_fun_evals or 100

        # Kernel selection
        self.kernel = kernel
        self.kernel_params = kernel_params or {}

        # Logging information
        self.log = {}
        self.log["negLnLike"] = []
        self.log["theta"] = []
        self.log["p"] = []
        self.log["Lambda"] = []

        self.logtheta_loglambda_p_ = None
        self.U_ = None
        self.X_ = None
        self.y_ = None
        self.negLnLike = None
        self.Psi_ = None
        if method not in ["interpolation", "regression", "reinterpolation"]:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation']")
        self.method = method
        self.return_ei = False
        self.return_std = False

        # Variable-type masks (set during fit)
        self.num_mask = None
        self.factor_mask = None
        self.int_mask = None
        self.ordered_mask = None

        # Nyström-specific attributes
        self.use_nystrom = use_nystrom
        self.nystrom_m = nystrom_m
        self.nystrom_seed = nystrom_seed
        self.landmark_idx_ = None
        self.X_m_ = None  # landmarks (m x k)
        self.W_chol_ = None  # chol(W) of K_mm
        self.M_chol_ = None  # chol(M) where M = W + (1/λ) C^T C
        self.C_ = None  # K_nm
        self.lambda_lin_ = None  # λ (linear scale)
        self.mu_ = None  # μ̂
        self.sigma2_ = None  # σ̂^2
        self.Rinv_one_ = None  # R^{-1} 1 (n,)
        self.Rinv_r_ = None  # R^{-1} r (n,)

    # --- Kernel dispatch ---

    def _correlation(self, D):
        """
        Dispatches to the selected kernel function.

        Args:
            D (np.ndarray): Distance matrix.

        Returns:
            np.ndarray: Correlation matrix.
        """
        if callable(self.kernel):
            return self.kernel(D, **self.kernel_params)
        elif self.kernel == "gauss":
            return gauss_kernel(D)
        elif self.kernel == "matern":
            nu = self.kernel_params.get("nu", 2.5)
            return matern_kernel(D, nu=nu)
        elif self.kernel == "exp":
            return exponential_kernel(D)
        elif self.kernel == "cubic":
            return cubic_kernel(D)
        elif self.kernel == "linear":
            return linear_kernel(D)
        elif self.kernel == "rq":
            alpha = self.kernel_params.get("alpha", 1.0)
            return rational_quadratic_kernel(D, alpha=alpha)
        elif self.kernel == "poly":
            degree = self.kernel_params.get("degree", 2)
            return poly_kernel(D, degree=degree)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    # -------- Basis correlation construction (identical to kriging.py) --------

    def _get_eps(self) -> float:
        """
        Computes a small epsilon value for numerical stability.
        Returns the square root of the machine epsilon.

        Returns:
            float: A small epsilon value.
        """
        eps = np.finfo(float).eps
        return np.sqrt(eps)

    def _set_variable_types(self) -> None:
        """
        Sets the variable types and their corresponding masks.
        Ensures that var_type has length k by repeating "num" if necessary. Sets the variable types for the class instance.
        This method sets the variable types for the class instance based
        on the `var_type` attribute. If the length of `var_type` is less
        than `k`, all variable types are forced to 'num' and a warning is logged.
        The method then creates Boolean masks for each variable
        type ('num', 'factor', 'int', 'ordered') using numpy arrays, e.g.,
        `num_mask = array([ True,  True])` if two numerical variables are present.

        Examples:
            >>> from spotpython.build import Kriging
                import numpy as np
                nat_X = np.array([[1, 2], [3, 4], [5, 6]])
                nat_y = np.array([1, 2, 3])
                var_type = ["num", "int", "float"]
                n_theta=2
                n_p=2
                S=Kriging(var_type=var_type, seed=124, n_theta=n_theta, n_p=n_p, optim_p=True)
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
        """
        if len(self.var_type) < self.k:
            self.var_type = ["num"] * self.k
        var_type_array = np.array(self.var_type)
        self.num_mask = var_type_array == "num"
        self.factor_mask = var_type_array == "factor"
        self.int_mask = var_type_array == "int"
        self.ordered_mask = np.isin(var_type_array, ["int", "num", "float"])

    def get_model_params(self) -> Dict[str, float]:
        """
        Returns the model parameters as a dictionary.

        Returns:
            Dict[str, float]: A dictionary containing model parameters.

        Examples:
            >>> import numpy as np
            >>> from spotpython.surrogate.kriging import Kriging
            >>> # Training data
            >>> X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            >>> y_train = np.array([0.1, 0.2, 0.3])
            >>> # Initialize and fit the Kriging model
            >>> model = Kriging()
            >>> model.fit(X_train, y_train)
            >>> # get theta values of the fitted model
            >>> X_values = model.get_model_params()["X"]
            >>> print("X values:", X_values)
        """
        return {
            "n": self.n,
            "k": self.k,
            "logtheta_loglambda_p_": self.logtheta_loglambda_p_,
            "U": self.U_,
            "X": self.X_,
            "y": self.y_,
            "negLnLike": self.negLnLike,
            "inf_Psi": getattr(self, "inf_Psi", None),
            "cnd_Psi": getattr(self, "cnd_Psi", None),
        }

    def _update_log(self) -> None:
        """
        If spot_writer is not None, this method writes the current values of
        negLnLike, theta, p (if optim_p is True),
        and Lambda (if method is not "interpolation") to the spot_writer object.

        Args:
            self (object): The Kriging object.

        Returns:
            None

        """
        self.log["negLnLike"] = append(self.log["negLnLike"], self.negLnLike)
        self.log["theta"] = append(self.log["theta"], self.theta)
        if self.optim_p:
            self.log["p"] = append(self.log["p"], self.p_val)
        if (self.method == "regression") or (self.method == "reinterpolation"):
            self.log["Lambda"] = append(self.log["Lambda"], self.Lambda)
        self.log_length = len(self.log["negLnLike"])
        if self.spot_writer is not None:
            negLnLike = float(self.negLnLike)
            self.spot_writer.add_scalar("spot_negLnLike", negLnLike, (self.counter or 0) + self.log_length)
            theta = np.array(self.theta, copy=True)
            self.spot_writer.add_scalars("spot_theta", {f"theta_{i}": theta[i] for i in range(self.n_theta)}, (self.counter or 0) + self.log_length)
            if (self.method == "regression") or (self.method == "reinterpolation"):
                Lambda = np.array(self.Lambda, copy=True)
                self.spot_writer.add_scalar("spot_Lambda", float(Lambda), (self.counter or 0) + self.log_length)
            if self.optim_p:
                p = np.array(self.p_val, copy=True)
                self.spot_writer.add_scalars("spot_p", {f"p_{i}": p[i] for i in range(self.n_p)}, (self.counter or 0) + self.log_length)
            self.spot_writer.flush()

    def _set_theta(self) -> None:
        self.n_theta = 1 if self.isotropic else self.k

    def _get_theta10_from_logtheta(self) -> np.ndarray:
        theta10 = np.power(10.0, self.theta)
        if self.n_theta == 1:
            theta10 = theta10 * np.ones(self.k)
        return theta10

    def _reshape_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, self.k)
        else:
            if X.shape[1] != self.k:
                if X.shape[0] == self.k:
                    X = X.T
                elif self.k == 1:
                    X = X.reshape(-1, 1)
                else:
                    raise ValueError(f"X has shape {X.shape}, expected (*, {self.k}).")
        return X

    # -------- Basis correlation construction (identical to kriging.py) --------

    def build_Psi(self) -> np.ndarray:
        """
        Constructs a new (n x n) correlation matrix Psi to reflect new data
        or a change in hyperparameters.
        This method uses `theta`, `p`, and coded `X` values to construct the
        correlation matrix as described in [Forrester et al., p.57].

        Notes:
            - Correlation follows the selected kernel:
              R = kernel(D), with D a weighted distance.
            - The code builds D as a sum of per-dimension distance contributions
              scaled by 10**theta (theta is stored in log10), then applies the kernel.
            - Returns only the upper triangle; the symmetric and diagonal parts
              are handled by the caller.

        Attributes:
            Psi (np.matrix): Correlation matrix Psi. Shape (n,n).
            cnd_Psi (float): Condition number of Psi.
            inf_Psi (bool): True if Psi is infinite, False otherwise.

        Raises:
            LinAlgError: If building Psi fails.

        Examples:
            >>> import numpy as np
            >>> from spotpython.surrogate.kriging import Kriging
            >>> # Training data
            >>> X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            >>> y_train = np.array([0.1, 0.2, 0.3])
            >>> # Fit the Kriging model with a Matern kernel
            >>> model = Kriging(kernel="matern", kernel_params={"nu": 1.5}).fit(X_train, y_train)
            >>> Psi = model.build_Psi()
            >>> print("Correlation matrix Psi:\n", Psi)
        """
        try:
            n, k = self.X_.shape
            theta10 = self._get_theta10_from_logtheta()

            Psi = np.zeros((n, n), dtype=np.float64)

            if self.ordered_mask.any():
                X_ordered = self.X_[:, self.ordered_mask]
                D_ordered = squareform(pdist(X_ordered, metric="sqeuclidean", w=theta10[self.ordered_mask]))
                Psi += D_ordered

            if self.factor_mask.any():
                X_factor = self.X_[:, self.factor_mask]
                D_factor = squareform(pdist(X_factor, metric=self.metric_factorial, w=theta10[self.factor_mask]))
                Psi += D_factor

            Psi = self._correlation(Psi)

            self.inf_Psi = np.isinf(Psi).any()
            self.cnd_Psi = cond(Psi)

            return np.triu(Psi, k=1)
        except LinAlgError as err:
            print("Building Psi failed. Error: %s, Type: %s", err, type(err))
            raise

    def _kernel_cross(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Cross-correlation matrix K(A,B) using the same distance definition as build_Psi/build_psi_vec.
        Returns kernel(D) with D composed from ordered and factor contributions.

        Args:
            A (np.ndarray): First set of points (m x k).
            B (np.ndarray): Second set of points (n x k).

        Returns:
            np.ndarray: Cross-correlation matrix K(A,B) of shape (m, n).

        Examples:
            >>> import numpy as np
            >>> from spotpython.surrogate.kriging import Kriging
            >>> X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            >>> y_train = np.array([0.1, 0.2, 0.3])
            >>> model = Kriging(kernel="poly", kernel_params={"degree": 3}).fit(X_train, y_train)
            >>> A = np.array([[0.0, 0.0], [1.0, 1.0]])
            >>> B = np.array([[0.5, 0.5], [1.5, 1.5]])
            >>> K_AB = model._kernel_cross(A, B)
            >>> print("Cross-correlation matrix K(A, B):\n", K_AB)
        """
        A = np.asarray(A)
        B = np.asarray(B)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if B.ndim == 1:
            B = B.reshape(1, -1)
        theta10 = self._get_theta10_from_logtheta()

        D = np.zeros((A.shape[0], B.shape[0]), dtype=float)
        if self.ordered_mask.any():
            Ao = A[:, self.ordered_mask]
            Bo = B[:, self.ordered_mask]
            D += cdist(Ao, Bo, metric="sqeuclidean", w=theta10[self.ordered_mask])
        if self.factor_mask.any():
            Af = A[:, self.factor_mask]
            Bf = B[:, self.factor_mask]
            D += cdist(Af, Bf, metric=self.metric_factorial, w=theta10[self.factor_mask])

        return self._correlation(D)

    def build_psi_vec(self, x: np.ndarray) -> np.ndarray:
        """
        Build the psi vector required for predictive methods.
        ψ(x) := [kernel(D(x, x_i))]_{i=1..n}, i.e., correlation between a new x and the training sites
        using the same D as for R.

        Args:
            x (ndarray): Point to calculate the psi vector for.

        Returns:
            np.ndarray: The psi vector.

        Examples:
            >>> import numpy as np
            >>> from spotpython.surrogate.kriging import Kriging
            >>> X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            >>> y_train = np.array([0.1, 0.2, 0.3])
            >>> model = Kriging(kernel="rq", kernel_params={"alpha": 2.0}).fit(X_train, y_train)
            >>> x_new = np.array([0.25, 0.25])
            >>> psi_vector = model.build_psi_vec(x_new)
            >>> print("Psi vector for new point:\n", psi_vector)
        """
        try:
            psi = self._kernel_cross(np.asarray(x), self.X_).ravel()
            return psi
        except np.linalg.LinAlgError as err:
            print("Building psi failed due to a linear algebra error: %s. Error type: %s", err, type(err))
            return np.zeros(self.X_.shape[0])

    # -------- Exact (Cholesky) likelihood --------

    def _likelihood_exact(self, x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Computes the negative of the concentrated log-likelihood for a given set
        of log(theta) parameters. Returns the negative log-likelihood, the correlation matrix Psi, and its Cholesky factor U.
        Negative concentrated log-likelihood (Forrester, Ch. 3):
          - Given R (here Psi with diagonal and nugget), μ = (1^T R^{-1} y)/(1^T R^{-1} 1),
            σ^2 = (r^T R^{-1} r)/n with r = y - 1·μ,
          - Concentrated −log L = (n/2) log(σ^2) + (1/2) log |R| (constants omitted).

        Args:
            x (np.ndarray):
                1D array of log(theta), log(Lambda) (if method is "regression" or "reinterpolation"), and p values (if optim_p is True).

        Returns:
            (float, np.ndarray, np.ndarray):
                (negLnLike, Psi, U) where:
                - negLnLike (float): The negative concentrated log-likelihood.
                - Psi (np.ndarray): The correlation matrix.
                - U (np.ndarray): The Cholesky factor (or None if ill-conditioned).

        Examples:
            >>> import numpy as np
                from spotpython.surrogate.kriging import Kriging
                # Training data
                X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
                y_train = np.array([0.1, 0.2, 0.3])
                # Fit the Kriging model
                model = Kriging().fit(X_train, y_train)
                log_theta = np.array([0.0, 0.0, -6.0]) # nugget: -6 => 10**(-6) = 1e-6
                negLnLike, Psi, U = model._likelihood_exact(log_theta)
                print("Negative Log-Likelihood:", negLnLike)
                print("Correlation matrix Psi:\n", Psi)
                print("Cholesky factor U (lower triangular):\n", U)
        """
        X = self.X_
        y = self.y_.flatten()
        self.theta = x[: self.n_theta]

        if (self.method == "regression") or (self.method == "reinterpolation"):
            lambda_ = 10.0 ** x[self.n_theta : self.n_theta + 1]
            if self.optim_p:
                self.p_val = x[self.n_theta + 1 : self.n_theta + 1 + self.n_p]
        elif self.method == "interpolation":
            lambda_ = self.eps
            if self.optim_p:
                self.p_val = x[self.n_theta : self.n_theta + self.n_p]
        else:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation'")

        n = X.shape[0]
        one = np.ones(n)

        Psi_up = self.build_Psi()
        Psi = Psi_up + Psi_up.T + np.eye(n) + np.eye(n) * lambda_

        try:
            U = np.linalg.cholesky(Psi)
        except LinAlgError:
            return self.penalty, Psi, None

        LnDetPsi = 2.0 * np.sum(np.log(np.abs(np.diag(U))))

        temp_y = np.linalg.solve(U, y)
        temp_one = np.linalg.solve(U, one)
        vy = np.linalg.solve(U.T, temp_y)
        vone = np.linalg.solve(U.T, temp_one)

        mu = (one @ vy) / (one @ vone)

        resid = y - one * mu
        tresid = np.linalg.solve(U, resid)
        tresid = np.linalg.solve(U.T, tresid)
        SigmaSqr = (resid @ tresid) / n

        negLnLike = (n / 2.0) * np.log(SigmaSqr) + 0.5 * LnDetPsi
        return negLnLike, Psi, U

    # -------- Nyström likelihood (low-rank) --------

    def _nystrom_setup(self) -> None:
        """
        Selects landmarks and builds K_mm (W) and K_nm (C) for the current theta.

        Examples:
            >>> import numpy as np
                from spotpython.surrogate.kriging import Kriging
                # Training data
                X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.25, 0.25], [0.75, 0.75]])
                y_train = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
                # Fit the Kriging model with Nyström approximation
                model = Kriging(use_nystrom=True, nystrom_m=3).fit(X_train, y_train)
                model._nystrom_setup()
                print("Landmark indices:", model.landmark_idx_)
                print("Landmark points (X_m_):\n", model.X_m_)
                print("Cholesky factor of W (W_chol_):\n", model.W_chol_)
        """
        n = self.n
        m = self.nystrom_m or max(10, min(n // 2, 300))
        m = min(m, n - 1) if n > 1 else 1
        rng = np.random.default_rng(self.nystrom_seed)
        self.landmark_idx_ = rng.choice(n, size=m, replace=False)
        self.X_m_ = self.X_[self.landmark_idx_]

        # W = K(X_m, X_m), C = K(X, X_m)
        W = self._kernel_cross(self.X_m_, self.X_m_)
        # Force symmetry and unit-diagonal structure exactly as full K would have
        W = (W + W.T) / 2.0
        np.fill_diagonal(W, 1.0)
        C = self._kernel_cross(self.X_, self.X_m_)
        self.C_ = C

        # Precompute chol(W) for logdet and stability checks
        try:
            self.W_chol_ = np.linalg.cholesky(W)
        except LinAlgError:
            # Fall back to adding tiny jitter and retry
            W_jit = W + 1e-12 * np.eye(W.shape[0])
            self.W_chol_ = np.linalg.cholesky(W_jit)

    def _woodbury_solve(self, v: np.ndarray) -> np.ndarray:
        """
        Applies R^{-1} v using Woodbury:
        R = λ I + C W^{-1} C^T  has inverse
        R^{-1} = (1/λ) I − (1/λ^2) C (W + (1/λ) C^T C)^{-1} C^T.
        Uses precomputed self.M_chol_ (chol of W + (1/λ) C^T C).

        Args:
            v (np.ndarray): Vector to solve against (n,).

        Returns:
            np.ndarray: Result of R^{-1} v (n,).

        Raises:
            RuntimeError: If required matrices not initialized
            ValueError: If input vector has wrong length

        Examples:
            >>> import numpy as np
                from spotpython.surrogate.kriging import Kriging
                # Training data
                X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.25, 0.25], [0.75, 0.75]])
                y_train = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
                # Fit the Kriging model with Nyström approximation
                model = Kriging(use_nystrom=True, nystrom_m=3).fit(X_train, y_train)
                model._nystrom_setup()
                v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
                Rinv_v = model._woodbury_solve(v)
                print("Result of R^{-1} v:\n", Rinv_v)
                # Note: Ensure that model.M_chol_ is computed before calling this method.
        """
        # Input validation and reshaping
        v = np.asarray(v).reshape(-1)  # Flatten to 1D
        if v.size != self.n:
            raise ValueError(f"Input vector has wrong length: {v.size} != {self.n}")

        if getattr(self, "C_", None) is None or getattr(self, "M_chol_", None) is None:
            raise RuntimeError("Required matrices not initialized. Call _nystrom_setup() first.")

        lam = float(self.lambda_lin_)
        C = self.C_

        # rhs = C^T v
        rhs = C.T @ v

        # Solve M α = rhs via two triangular solves with M_chol
        temp = np.linalg.solve(self.M_chol_, rhs)
        alpha = np.linalg.solve(self.M_chol_.T, temp)

        # Apply Woodbury formula: (1/λ)v - (1/λ^2)C α
        result = (v - C @ alpha) / lam
        return result

    def _likelihood_nystrom(self, x: np.ndarray) -> Tuple[float, None, None]:
        """
        Low-rank (Nyström) concentrated likelihood using matrix determinant lemma and Woodbury.
        Returns (negLnLike, None, None) to match the exact signature.

        Args:
            x (np.ndarray): Parameter vector containing [log(theta)..., log10(lambda), p...].
                Length depends on method and optim_p setting.

        Returns:
            Tuple[float, None, None]: (negative log likelihood, None, None)

        Notes:
            1. Build K_mm (W) and K_nm (C) for current theta.
            2. Compute M = W + (1/λ) C^T C and its Cholesky.
            3. Compute log|R| via determinant lemma.
            4. Compute R^{-1} y and R^{-1} 1 via Woodbury.
            5. Compute μ̂ and σ̂^2.
            6. Return negative concentrated log-likelihood.
            7. If σ̂^2 ≤ 0 or non-finite, return penalty.

        Raises:
            ValueError: If input parameter vector has wrong length.

        Examples:
            >>> import numpy as np
                from spotpython.surrogate.kriging import Kriging
                # Training data
                X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.25, 0.25], [0.75, 0.75]])
                y_train = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
                # Fit the Kriging model with Nyström approximation
                model = Kriging(use_nystrom=True, nystrom_m=3).fit(X_train, y_train)
                log_theta = np.array([0.0, 0.0, -6.0]) # nugget: -6 => 10**(-6) = 1e-6
                negLnLike, _, _ = model._likelihood_nystrom(log_theta)
                print("Negative Log-Likelihood (Nyström):", negLnLike)
        """
        # Input validation
        x = np.asarray(x, dtype=float)
        expected_dim = self.n_theta
        if self.method in ["regression", "reinterpolation"]:
            expected_dim += 1  # Add lambda parameter
        if self.optim_p:
            expected_dim += self.k  # Add p parameters

        if x.size != expected_dim:
            return self.penalty, None, None  # Return penalty for wrong dimensions

        # Get data
        X = self.X_
        y = self.y_.flatten()
        n = X.shape[0]

        # Extract parameters
        self.theta = x[: self.n_theta]

        # Handle lambda based on method
        if self.method in ["regression", "reinterpolation"]:
            lambda_idx = self.n_theta
            try:
                self.lambda_lin_ = float(10.0 ** x[lambda_idx])
            except (IndexError, TypeError):
                return self.penalty, None, None

            if self.optim_p:
                self.p_val = x[self.n_theta + 1 : self.n_theta + 1 + self.n_p]
        else:  # interpolation
            self.lambda_lin_ = float(self.eps)
            if self.optim_p:
                self.p_val = x[self.n_theta : self.n_theta + self.n_p]

        # Build Nyström structures for current theta
        self._nystrom_setup()
        C = self.C_
        W_chol = self.W_chol_
        m = W_chol.shape[0]  # number of landmarks
        one = np.ones(n)

        # Compute W = W_chol @ W_chol.T and M = W + (1/λ)C^TC
        W = W_chol @ W_chol.T
        CtC = C.T @ C
        M = W + CtC / self.lambda_lin_

        # Get Cholesky of M with jitter if needed
        try:
            self.M_chol_ = np.linalg.cholesky(M)
        except LinAlgError:
            M_jit = M + 1e-10 * np.eye(m)
            try:
                self.M_chol_ = np.linalg.cholesky(M_jit)
            except LinAlgError:
                return self.penalty, None, None

        # Compute log|R| via matrix determinant lemma
        logdetW = 2.0 * np.sum(np.log(np.abs(np.diag(W_chol))))
        logdetM = 2.0 * np.sum(np.log(np.abs(np.diag(self.M_chol_))))
        LnDetR = n * np.log(self.lambda_lin_) + (logdetM - logdetW)

        # Compute R^{-1}y and R^{-1}1 via Woodbury
        try:
            Rinv_y = self._woodbury_solve(y)
            Rinv_one = self._woodbury_solve(one)
        except (ValueError, LinAlgError):
            return self.penalty, None, None

        # Compute mu_hat and sigma2_hat
        mu = (one @ Rinv_y) / (one @ Rinv_one)
        r = y - one * mu
        Rinv_r = self._woodbury_solve(r)
        SigmaSqr = (r @ Rinv_r) / n

        # Check validity
        if SigmaSqr <= 0 or not np.isfinite(SigmaSqr) or not np.isfinite(LnDetR):
            return self.penalty, None, None

        # Compute negative concentrated log likelihood
        negLnLike = (n / 2.0) * np.log(SigmaSqr) + 0.5 * LnDetR

        if not np.isfinite(negLnLike):
            return self.penalty, None, None

        return negLnLike, None, None

    # -------- Unified likelihood --------

    def likelihood(self, x: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Computes the negative concentrated log-likelihood.
        Exact (Cholesky) if use_nystrom is False; otherwise Nyström.

        Args:
            x (np.ndarray): Parameter vector containing [log(theta)..., log10(lambda), p...].
                Length depends on method and optim_p setting.

        Returns:
            Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
                (negLnLike, Psi, U) where:
                - negLnLike (float): The negative concentrated log-likelihood.
                - Psi (np.ndarray or None): The correlation matrix (None if Nyström).
                - U (np.ndarray or None): The Cholesky factor (None if Nyström).

        Examples:
            >>> import numpy as np
                from spotpython.surrogate.kriging import Kriging
                # Training data
                X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
                y_train = np.array([0.1, 0.2, 0.3])
                # Fit the Kriging model
                model = Kriging().fit(X_train, y_train)
                log_theta = np.array([0.0, 0.0, -6.0]) # nugget: -6 => 10**(-6) = 1e-6
                negLnLike, Psi, U = model.likelihood(log_theta)
                print("Negative Log-Likelihood:", negLnLike)
                print("Correlation matrix Psi:\n", Psi)
                print("Cholesky factor U (lower triangular):\n", U)
        """
        if self.use_nystrom:
            return self._likelihood_nystrom(x)
        else:
            return self._likelihood_exact(x)

    # -------- Fit / optimize --------

    def max_likelihood(self, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        """
        Hyperparameter estimation via maximum likelihood (Forrester, Ch. 3).

        Args:
            bounds (List[Tuple[float, float]]): Bounds for the optimization variables.
        Returns:
            Tuple[np.ndarray, float]: (optimal parameters, optimal negative log-likelihood)
        Examples:
            >>> import numpy as np
                from spotpython.surrogate.kriging import Kriging
                # Training data
                X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
                y_train = np.array([0.1, 0.2, 0.3])
                # Fit the Kriging model
                model = Kriging().fit(X_train, y_train)
                bounds = [(-3, 3), (-3, 3), (-6, 0)] # Example bounds for log(theta) and log10(lambda)
                optimal_params, optimal_negLnLike = model.max_likelihood(bounds)
                print("Optimal parameters:", optimal_params)
                print("Optimal Negative Log-Likelihood:", optimal_negLnLike)
        """

        def objective(logtheta_loglambda_p_: np.ndarray) -> float:
            neg_ln_like, _, _ = self.likelihood(logtheta_loglambda_p_)
            return float(neg_ln_like)

        result = differential_evolution(func=objective, bounds=bounds, seed=self.seed)
        return result.x, result.fun

    def fit(self, X: np.ndarray, y: np.ndarray, bounds: Optional[List[Tuple[float, float]]] = None) -> "Kriging":
        """
        Fits the Kriging model. Public behavior matches the baseline class.
        Nyström path stores additional solver state for fast prediction.

        Args:
            X (np.ndarray): Training input data of shape (n_samples, n_features).
            y (np.ndarray): Training output data of shape (n_samples,).
            bounds (Optional[List[Tuple[float, float]]]): Bounds for hyperparameter optimization.
                If None, default bounds are used.

        Returns:
            Kriging: The fitted Kriging model instance.

        Examples:
            >>> import numpy as np
                from spotpython.surrogate.kriging import Kriging
                # Training data
                X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
                y_train = np.array([0.1, 0.2, 0.3])
                # Fit the Kriging model
                model = Kriging().fit(X_train, y_train)
                print("Fitted theta:", model.theta)
                print("Fitted Lambda:", model.Lambda)
                print("Fitted p:", model.p_val)
                # Print negative log-likelihood at optimum
                print("Negative Log-Likelihood at optimum:", model.negLnLike)
                # Print other relevant information
                print("Fitted hyperparameters:", model.logtheta_loglambda_p_)
                print("Fitted noise variance:", model.sigma2_)
                print("Fitted mu:", model.mu_)
                print("Fitted hyperparameters:", model.logtheta_loglambda_p_)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # --- Explicit shape checks BEFORE flattening y ---
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}")

        y = y.flatten()

        self.X_ = X
        self.y_ = y
        self.n, self.k = self.X_.shape
        self._set_variable_types()
        if self.n_theta is None:
            self._set_theta()
        self.min_X = np.min(self.X_, axis=0)
        self.max_X = np.max(self.X_, axis=0)

        # Bounds: [θ] for interpolation; [θ, λ] for regression/reinterpolation; [+ p] if optim_p
        if bounds is None:
            if self.method == "interpolation":
                bounds = [(self.min_theta, self.max_theta)] * self.k
            else:
                bounds = [(self.min_theta, self.max_theta)] * self.k + [(self.min_Lambda, self.max_Lambda)]
        if self.optim_p:
            n_p = self.n_p if hasattr(self, "n_p") else self.k
            bounds += [(self.min_p, self.max_p)] * n_p

        # Optimize concentrated likelihood
        self.logtheta_loglambda_p_, _ = self.max_likelihood(bounds)

        # Store parameters on log10 scale; convert to linear only where needed
        self.theta = self.logtheta_loglambda_p_[: self.n_theta]
        if (self.method == "regression") or (self.method == "reinterpolation"):
            self.Lambda = self.logtheta_loglambda_p_[self.n_theta : self.n_theta + 1]
            if self.optim_p:
                self.p_val = self.logtheta_loglambda_p_[self.n_theta + 1 : self.n_theta + 1 + self.n_p]
        elif self.method == "interpolation":
            self.Lambda = None
            if self.optim_p:
                self.p_val = self.logtheta_loglambda_p_[self.n_theta : self.n_theta + self.n_p]
        else:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation'")

        # Finalize model state at optimum:
        if self.use_nystrom:
            # Prepare Nyström state at the optimum
            if (self.method == "regression") or (self.method == "reinterpolation"):
                self.lambda_lin_ = float(10.0**self.Lambda)
            else:
                self.lambda_lin_ = float(self.eps)

            # Build W, C with final theta; store M chol
            self._nystrom_setup()
            C = self.C_
            W = self.W_chol_ @ self.W_chol_.T
            n = self.n
            m = W.shape[0]
            one = np.ones(n)
            yv = self.y_.flatten()

            M = W + (C.T @ C) / self.lambda_lin_
            try:
                self.M_chol_ = np.linalg.cholesky(M)
            except LinAlgError:
                self.M_chol_ = np.linalg.cholesky(M + 1e-12 * np.eye(m))

            # Precompute R^{-1}1, R^{-1}y, μ, σ^2, R^{-1}r
            Rinv_one = self._woodbury_solve(one)
            Rinv_y = self._woodbury_solve(yv)
            mu = (one @ Rinv_y) / (one @ Rinv_one)
            r = yv - one * mu
            Rinv_r = Rinv_y - Rinv_one * mu
            sigma2 = (r @ Rinv_r) / n

            # Store
            self.mu_ = float(mu)
            self.sigma2_ = float(max(0.0, sigma2))
            self.Rinv_one_ = Rinv_one
            self.Rinv_r_ = Rinv_r

            # Compute negLnLike for logging using Nyström
            negLnLike, _, _ = self._likelihood_nystrom(self.logtheta_loglambda_p_)
            self.negLnLike = float(negLnLike)
            self.Psi_, self.U_ = None, None
        else:
            # Exact final Psi and U
            self.negLnLike, self.Psi_, self.U_ = self._likelihood_exact(self.logtheta_loglambda_p_)

        self._update_log()
        return self

    # -------- Prediction --------

    def _pred_exact(self, x: np.ndarray) -> Tuple[float, float, Optional[float]]:
        """
        Exact prediction using Cholesky state:
                f̂(x) = μ̂ + ψ(x)^T R^{-1} r
                s^2(x) = σ̂^2 [1 + λ − ψ(x)^T R^{-1} ψ(x)]

        Args:
            x (np.ndarray): Point to predict at.

        Returns:
            Tuple[float, float, Optional[float]]: (f, s, ExpImp) where:
                - f (float): Predicted mean.
                - s (float): Predicted standard deviation.
                - ExpImp (Optional[float]): Expected improvement (log10 scale) if return_ei is True; otherwise None.

        Examples:
            >>> import numpy as np
                from spotpython.surrogate.kriging import Kriging
                # Training data
                X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
                y_train = np.array([0.1, 0.2, 0.3])
                # Fit the Kriging model
                model = Kriging().fit(X_train, y_train)
                x_new = np.array([0.25, 0.25])
                f, s, exp_imp = model._pred_exact(x_new)
                print("Predicted mean f:", f)
                print("Predicted standard deviation s:", s)
                if exp_imp is not None:
                    print("Expected Improvement (log10 scale):", exp_imp)
        """
        y = self.y_.flatten()

        if (self.method == "regression") or (self.method == "reinterpolation"):
            self.theta = self.logtheta_loglambda_p_[: self.n_theta]
            lambda_ = 10.0 ** self.logtheta_loglambda_p_[self.n_theta : self.n_theta + 1]
            if self.optim_p:
                self.p_val = self.logtheta_loglambda_p_[self.n_theta + 1 : self.n_theta + 1 + self.n_p]
        elif self.method == "interpolation":
            self.theta = self.logtheta_loglambda_p_[: self.n_theta]
            lambda_ = self.eps
            if self.optim_p:
                self.p_val = self.logtheta_loglambda_p_[self.n_theta : self.n_theta + self.n_p]
        else:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation'")

        U = self.U_
        n = self.X_.shape[0]
        one = np.ones(n)

        y_tilde = np.linalg.solve(U, y)
        y_tilde = np.linalg.solve(U.T, y_tilde)
        one_tilde = np.linalg.solve(U, one)
        one_tilde = np.linalg.solve(U.T, one_tilde)
        mu = (one @ y_tilde) / (one @ one_tilde)

        resid = y - one * mu
        resid_tilde = np.linalg.solve(U, resid)
        resid_tilde = np.linalg.solve(U.T, resid_tilde)

        psi = self.build_psi_vec(x)

        if (self.method == "interpolation") or (self.method == "regression"):
            SigmaSqr = (resid @ resid_tilde) / n
            psi_tilde = np.linalg.solve(U, psi)
            psi_tilde = np.linalg.solve(U.T, psi_tilde)
            SSqr = SigmaSqr * (1 + lambda_ - psi @ psi_tilde)
        else:
            Psi_adjusted = self.Psi_ - np.eye(n) * lambda_ + np.eye(n) * self.eps
            SigmaSqr = (resid @ np.linalg.solve(U.T, np.linalg.solve(U, Psi_adjusted @ resid_tilde))) / n
            Uint = np.linalg.cholesky(Psi_adjusted)
            psi_tilde = np.linalg.solve(Uint, psi)
            psi_tilde = np.linalg.solve(Uint.T, psi_tilde)
            SSqr = SigmaSqr * (1 - psi @ psi_tilde)

        s = float(np.abs(SSqr) ** 0.5)
        f = float(mu + psi @ resid_tilde)

        if self.return_ei:
            yBest = np.min(y)
            EITermOne = (yBest - f) * (0.5 + 0.5 * erf((1 / np.sqrt(2)) * ((yBest - f) / (s + self.eps))))
            EITermTwo = (s + self.eps) * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((yBest - f) ** 2 / ((s + self.eps) ** 2)))
            ExpImp = np.log10(EITermOne + EITermTwo + self.eps)
            return f, s, float(-ExpImp)
        else:
            return f, s, None

    def _pred_nystrom(self, x: np.ndarray) -> Tuple[float, float, Optional[float]]:
        """
        Prediction using Nyström state:
          f̂(x) = μ̂ + ψ(x)^T R^{-1} r
          s^2(x) = σ̂^2 [1 + λ − ψ(x)^T R^{-1} ψ(x)]

        Args:
            x (np.ndarray): Point to predict at.

        Returns:
            Tuple[float, float, Optional[float]]: (f, s, ExpImp) where:
                - f (float): Predicted mean.
                - s (float): Predicted standard deviation.
                - ExpImp (Optional[float]): Expected improvement (log10 scale) if return_ei is True; otherwise None.

        Examples:
            >>> import numpy as np
                from spotpython.surrogate.kriging import Kriging
                # Training data
                X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.25, 0.25], [0.75, 0.75]])
                y_train = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
                # Fit the Kriging model with Nyström approximation
                model = Kriging(use_nystrom=True, nystrom_m=3).fit(X_train, y_train)
                x_new = np.array([0.5, 0.5])
                f, s, exp_imp = model._pred_nystrom(x_new)
                print("Predicted mean f:", f)
                print("Predicted standard deviation s:", s)
                if exp_imp is not None:
                    print("Expected Improvement (log10 scale):", exp_imp)
        """
        # Ensure state is available
        if self.C_ is None or self.M_chol_ is None or self.Rinv_r_ is None:
            # Fallback to exact if state missing
            return self._pred_exact(x)

        psi = self.build_psi_vec(x)
        # Apply R^{-1} to psi via Woodbury
        z = self._woodbury_solve(psi)
        f = float(self.mu_ + psi @ self.Rinv_r_)
        s2 = float(self.sigma2_ * (1.0 + self.lambda_lin_ - psi @ z))
        s = float(np.sqrt(max(0.0, s2)))

        if self.return_ei:
            y = self.y_.flatten()
            yBest = np.min(y)
            EITermOne = (yBest - f) * (0.5 + 0.5 * erf((1 / np.sqrt(2)) * ((yBest - f) / (s + self.eps))))
            EITermTwo = (s + self.eps) * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((yBest - f) ** 2 / ((s + self.eps) ** 2)))
            ExpImp = np.log10(EITermOne + EITermTwo + self.eps)
            return f, s, float(-ExpImp)
        else:
            return f, s, None

    def _pred(self, x: np.ndarray) -> Tuple[float, float, Optional[float]]:
        """
        Computes a single-point Kriging prediction (exact or Nyström).

        Args:
            x (np.ndarray): Point to predict at.

        Returns:
            Tuple[float, float, Optional[float]]: (f, s, ExpImp) where:
                - f (float): Predicted mean.
                - s (float): Predicted standard deviation.
                - ExpImp (Optional[float]): Expected improvement (log10 scale) if return_ei is True; otherwise None.

        Examples:
            >>> import numpy as np
                from spotpython.surrogate.kriging import Kriging
                import matplotlib.pyplot as plt

                # 1D Training data
                X_train = np.array([[0.0], [0.3], [0.6], [1.0]])
                y_train = np.sin(2 * np.pi * X_train).ravel()  # Example: noisy sine

                # Fit the Kriging model (exact)
                model = Kriging().fit(X_train, y_train)

                # Fit the Kriging model (Nyström)
                model_nystrom = Kriging(use_nystrom=True, nystrom_m=2).fit(X_train, y_train)

                # Predict on a grid
                x_grid = np.linspace(0, 1, 200).reshape(-1, 1)
                y_pred_exact = model.predict(x_grid)
                y_pred_nystrom = model_nystrom.predict(x_grid)

                # Plot
                plt.figure(figsize=(8, 4))
                plt.plot(x_grid, y_pred_exact, label='Exact Kriging')
                plt.plot(x_grid, y_pred_nystrom, '--', label='Nyström Kriging')
                plt.scatter(X_train, y_train, color='red', label='Training Data')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                plt.title('Kriging and Nyström Kriging (1D Example)')
                plt.grid()
                plt.show()

        """
        if self.use_nystrom:
            return self._pred_nystrom(x)
        else:
            return self._pred_exact(x)

    def predict(self, X: np.ndarray, return_std=False, return_val: str = "y") -> np.ndarray:
        """
        Batch prediction wrapper around _pred, identical public behavior to baseline.

        Args:
            X (np.ndarray): Points to predict at, shape (n_samples, n_features).
            return_std (bool): Whether to return standard deviations.
            return_val (str): What to return: "y" (default), "s", "ei", or "all".

        Returns:
            np.ndarray: Predictions, standard deviations, and/or expected improvements.

        Examples:
            >>> import numpy as np
                from spotpython.surrogate.kriging import Kriging
                # Training data
                X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
                y_train = np.array([0.1, 0.2, 0.3])
                # Fit the Kriging model
                model = Kriging().fit(X_train, y_train)
                # Predict at new points
                X_new = np.array([[0.25, 0.25], [0.75, 0.75]])
                predictions = model.predict(X_new)
                print("Predictions:", predictions)
                # Predict with standard deviations
                predictions, std_devs = model.predict(X_new, return_std=True)
                print("Predictions:", predictions)
                print("Standard Deviations:", std_devs)
                # Predict expected improvements
                eis = model.predict(X_new, return_val="ei")
                print("Expected Improvements (log10 scale):", eis)
        """
        self.return_std = return_std
        X = self._reshape_X(X)

        if return_std:
            self.return_std = True
            predictions, std_devs = zip(*[self._pred(x_i)[:2] for x_i in X])
            return np.array(predictions), np.array(std_devs)
        if return_val == "s":
            self.return_std = True
            predictions, std_devs = zip(*[self._pred(x_i)[:2] for x_i in X])
            return np.array(std_devs)
        elif return_val == "all":
            self.return_std = True
            self.return_ei = True
            predictions, std_devs, eis = zip(*[self._pred(x_i) for x_i in X])
            return np.array(predictions), np.array(std_devs), np.array(eis)
        elif return_val == "ei":
            self.return_ei = True
            predictions, eis = zip(*[(self._pred(x_i)[0], self._pred(x_i)[2]) for x_i in X])
            return np.array(eis)
        else:
            predictions = [self._pred(x_i)[0] for x_i in X]
            return np.array(predictions)

    # -------- Plot (same behavior as baseline) --------

    def plot(self, i: int = 0, j: int = 1, show: Optional[bool] = True, add_points: bool = True) -> None:
        """
        Plots the Kriging model. For 1D, plots the prediction curve. For 2D, creates a contour plot.
        Uses Nyström prediction if enabled.

        Args:
            i (int): Index of the first variable to plot (default is 0).
            j (int): Index of the second variable to plot (default is 1).
            show (Optional[bool]): Whether to display the plot immediately (default is True).
            add_points (bool): Whether to add training data points to the plot (default is True).

        Returns:
            None

        Examples:
            >>> import numpy as np
                from spotpython.surrogate.kriging import Kriging
                import matplotlib.pyplot as plt
                # 1D Training data
                X_train = np.array([[0.0], [0.3], [0.6], [1.0]])
                y_train = np.sin(2 * np.pi * X_train).ravel()  # Example: noisy sine
                # Fit the Kriging model with Nyström approximation
                model = Kriging(use_nystrom=True, nystrom_m=2).fit(X_train, y_train)
                # Plot the model
                model.plot()
        """
        if self.k == 1:
            n_grid = 100
            x = linspace(self.min_X[0], self.max_X[0], num=n_grid).reshape(-1, 1)
            y = self.predict(x)
            plt.figure()
            plt.plot(x.ravel(), y, "k")
            if add_points:
                plt.plot(self.X_[:, 0], self.y_, "ro")
            plt.grid()
            if show:
                plt.show()
        else:
            plotkd(model=self, X=self.X_, y=self.y_, i=i, j=j, show=show, var_type=self.var_type, add_points=True)
