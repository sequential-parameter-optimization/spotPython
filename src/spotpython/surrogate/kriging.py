import numpy as np
from numpy.linalg import LinAlgError
from typing import Dict, Tuple, List, Optional
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.special import erf
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid, array, append
import pylab
from numpy import ravel
from spotpython.utils.aggregate import aggregate_mean_var
import logging

logger = logging.getLogger(__name__)
# configure the handler and formatter as needed
py_handler = logging.FileHandler(f"{__name__}.log", mode="w")
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
# add formatter to the handler
py_handler.setFormatter(py_formatter)
# add handler to the logger
logger.addHandler(py_handler)


class Kriging(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible Kriging model class for regression tasks.
    Provides methods for likelihood evaluation, predictions, and hyperparameter optimization.

    Attributes:
        eps (float): A small regularization term to reduce ill-conditioning.
        penalty (float): The penalty value used if the correlation matrix is ill-conditioned.
        logtheta_lambda_ (np.ndarray): Best-fit log(theta) parameters from fit().
        U_ (np.ndarray): The Cholesky factor of the correlation matrix after fit().
        X_ (np.ndarray): The training input data (n x d).
        y_ (np.ndarray): The training target values (n,).
    """

    def __init__(
        self,
        eps: float = None,
        penalty: float = 1e4,
        method="regression",
        noise: bool = False,
        var_type: List[str] = ["num"],
        name: str = "Kriging",
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
        max_Lambda: float = 1.0,
        log_level: int = 50,
        spot_writer=None,
        counter=None,
        metric_factorial="canberra",
        **kwargs,
    ):
        """
        Initializes the Kriging model.

        Args:
            eps (float, optional):
                Small number added to the diagonal of the correlation matrix to reduce
                ill-conditioning. Defaults to the square root of machine epsilon.
                Only used if method is "interpolation". Otherwise, if method is "regression" or "reinterpolation", eps is replaced by the
                lambda_ parameter. Defaults to None.
            penalty (float, optional):
                Large negative log-likelihood assigned if the correlation matrix is
                not positive-definite. Defaults to 1e4.
            method (str, optional):
                The type how the model uis fitted. Can be "interpolation", "regression", or "reinterpolation". Defaults to "regression".
        """
        if eps is None:
            self.eps = self._get_eps()
        else:
            # check if eps is positive
            if eps <= 0:
                raise ValueError("eps must be positive")
            self.eps = eps
        self.penalty = penalty

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
        if self.model_fun_evals is None:
            self.model_fun_evals = 100

        # Logging information
        self.log = {}
        self.log["negLnLike"] = []
        self.log["theta"] = []
        self.log["p"] = []
        self.log["Lambda"] = []

        self.logtheta_lambda_ = None
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

    def _get_eps(self) -> float:
        """
        Returns the square root of the machine epsilon.
        """
        eps = np.finfo(float).eps
        return np.sqrt(eps)

    def get_model_params(self) -> Dict[str, float]:
        """
        Get the model parameters (in addition to sklearn's get_params method).

        This method is NOT required for scikit-learn compatibility.

        Returns:
            dict: Parameter names not included in get_params() mapped to their values.
        """
        return {"log_theta_lambda": self.logtheta_lambda_, "U": self.U_, "X": self.X_, "y": self.y_, "negLnLike": self.negLnLike}

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
            self.log["p"] = append(self.log["p"], self.p)
        if (self.method == "regression") or (self.method == "reinterpolation"):
            self.log["Lambda"] = append(self.log["Lambda"], self.Lambda)
        # get the length of the log
        self.log_length = len(self.log["negLnLike"])
        if self.spot_writer is not None:
            negLnLike = self.negLnLike.copy()
            self.spot_writer.add_scalar("spot_negLnLike", negLnLike, self.counter + self.log_length)
            # add the self.n_theta theta values to the writer with one key "theta",
            # i.e, the same key for all theta values
            theta = self.theta.copy()
            self.spot_writer.add_scalars("spot_theta", {f"theta_{i}": theta[i] for i in range(self.n_theta)}, self.counter + self.log_length)
            if (self.method == "regression") or (self.method == "reinterpolation"):
                Lambda = self.Lambda.copy()
                self.spot_writer.add_scalar("spot_Lambda", Lambda, self.counter + self.log_length)
            if self.optim_p:
                p = self.p.copy()
                self.spot_writer.add_scalars("spot_p", {f"p_{i}": p[i] for i in range(self.n_p)}, self.counter + self.log_length)
            self.spot_writer.flush()

    def fit(self, X: np.ndarray, y: np.ndarray, bounds: Optional[List[Tuple[float, float]]] = None) -> "Kriging":
        """
        Fits the Kriging model to training data X and y. This method is compatible
        with scikit-learn and uses differential evolution to optimize the hyperparameters
        (log(theta)).

        Args:
            X (np.ndarray):
                Training input data of shape (n_samples, n_features).
            y (np.ndarray):
                Target values of shape (n_samples,) or (n_samples, 1).
            bounds (Optional[List[Tuple[float, float]]]):
                Bounds for each dimension of log(theta). If None, defaults to
                [(-3, 2)] * n_features for interpolation, or
                [(-3, 2)] * n_features + [(-6, 0)] for regression/reinterpolation.

        Returns:
            Kriging:
                The fitted Kriging model instance (self).

        Raises:
            ValueError: If input data has invalid shape or contains invalid values.
            RuntimeError: If optimization fails or correlation matrix is singular.

        Examples:
            >>> import numpy as np
            >>> from spotpython.surrogate.kriging import Kriging
            >>> # Training data
            >>> X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            >>> y_train = np.array([0.1, 0.2, 0.3])
            >>> # Initialize and fit the Kriging model
            >>> model = Kriging()
            >>> model.fit(X_train, y_train)
            >>> print("Fitted log(theta):", model.logtheta_lambda_)
        """
        # Input validation and preprocessing
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()

        # Validate input shapes
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D array with shape {X.shape}")

        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array, got {y.ndim}D array with shape {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X ({X.shape[0]}) must match number of samples in y ({y.shape[0]})")

        # Check for minimum number of samples
        if X.shape[0] < 2:
            raise ValueError("At least 2 samples are required for fitting")

        # Check for invalid values
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values (NaN or inf)")

        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values (NaN or inf)")

        # Store training data FIRST before aggregation
        self.X_ = X.copy()  # Create a copy to avoid external modifications
        self.y_ = y.copy()
        self.n, self.k = X.shape

        # Calculate and store min and max of X for plotting and validation
        self.min_X = np.min(self.X_, axis=0)
        self.max_X = np.max(self.X_, axis=0)

        # Aggregate data for duplicates (if any) - NOW self.X_ and self.y_ are available
        try:
            _, aggregated_mean_y, _ = aggregate_mean_var(X=self.X_, y=self.y_)
            self.aggregated_mean_y = np.copy(aggregated_mean_y)
        except Exception as e:
            raise RuntimeError(f"Failed to aggregate training data: {e}")

        # Check for duplicate rows (which can cause numerical issues)
        if X.shape[0] > 1:
            unique_rows = np.unique(X, axis=0)
            if len(unique_rows) != X.shape[0] and self.method == "interpolation":
                logger.warning(f"Found {X.shape[0] - len(unique_rows)} duplicate rows in X. " "This may cause numerical issues with interpolation method.")

        # Check for zero variance in any dimension
        if np.any(self.max_X - self.min_X == 0):
            zero_var_dims = np.where(self.max_X - self.min_X == 0)[0]
            logger.warning(f"Zero variance detected in dimensions {zero_var_dims}. " "This may cause numerical issues.")

        # Set optimization bounds
        if bounds is None:
            if self.method == "interpolation":
                bounds = [(self.min_theta, self.max_theta)] * self.k
            else:
                # regression and reinterpolation use lambda_ as well
                bounds = [(self.min_theta, self.max_theta)] * self.k + [(np.log10(self.min_Lambda), np.log10(self.max_Lambda))]
        else:
            # Validate user-provided bounds
            expected_length = self.k if self.method == "interpolation" else self.k + 1
            if len(bounds) != expected_length:
                raise ValueError(f"bounds must have length {expected_length} for method '{self.method}', " f"got {len(bounds)}")

            # Validate individual bounds
            for i, (low, high) in enumerate(bounds):
                if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
                    raise ValueError(f"bounds[{i}] must contain numeric values")
                if low >= high:
                    raise ValueError(f"bounds[{i}]: lower bound ({low}) must be less than upper bound ({high})")

        # Optimize hyperparameters
        try:
            logger.info(f"Starting hyperparameter optimization with bounds: {bounds}")
            self.logtheta_lambda_, final_likelihood = self.max_likelihood(bounds)
            logger.info(f"Optimization completed. Final likelihood: {final_likelihood}")
        except Exception as e:
            raise RuntimeError(f"Hyperparameter optimization failed: {e}")

        # Validate optimization results
        if not np.all(np.isfinite(self.logtheta_lambda_)):
            raise RuntimeError("Optimization resulted in non-finite hyperparameters")

        # Extract and store theta and Lambda parameters
        if (self.method == "regression") or (self.method == "reinterpolation"):
            self.theta = self.logtheta_lambda_[:-1]
            self.Lambda = self.logtheta_lambda_[-1]
        else:
            self.theta = self.logtheta_lambda_
            self.Lambda = None

        # Store p for future use (currently fixed at 2)
        self.p = 2

        # Compute final correlation matrix and validate
        try:
            self.negLnLike, self.Psi_, self.U_ = self.likelihood(self.logtheta_lambda_)

            # Check if correlation matrix is well-conditioned
            if self.U_ is None:
                raise RuntimeError("Failed to compute Cholesky decomposition of correlation matrix")

            # Check condition number
            if hasattr(self, "Psi_") and self.Psi_ is not None:
                try:
                    cond_num = np.linalg.cond(self.Psi_)
                    if cond_num > 1e12:
                        logger.warning(f"Correlation matrix is ill-conditioned (condition number: {cond_num:.2e})")
                except np.linalg.LinAlgError:
                    logger.warning("Could not compute condition number of correlation matrix")

        except Exception as e:
            raise RuntimeError(f"Failed to compute final correlation matrix: {e}")

        # Final validation
        if not np.isfinite(self.negLnLike):
            raise RuntimeError("Final likelihood is not finite")

        # Update logging information
        try:
            self._update_log()
        except Exception as e:
            logger.warning(f"Failed to update log: {e}")

        # Log fitting summary
        logger.info("Kriging model fitted successfully:")
        logger.info(f"  - Method: {self.method}")
        logger.info(f"  - Training samples: {self.n}")
        logger.info(f"  - Features: {self.k}")
        logger.info(f"  - Final negative log-likelihood: {self.negLnLike:.6f}")
        logger.info(f"  - Theta parameters: {self.theta}")
        if self.Lambda is not None:
            logger.info(f"  - Lambda parameter: {self.Lambda:.6f}")

        return self

    def predict(self, X: np.ndarray, return_std=False, return_val: str = "y") -> np.ndarray:
        """
        Predicts the Kriging response at a set of points X. This method is compatible
        with scikit-learn and returns predictions for the input points.

        Args:
            X (np.ndarray):
                Array of shape (n_samples, n_features) containing the points at which
                to predict the Kriging response.
            return_std (bool, optional):
                If True, returns the standard deviation of the predictions as well.
                Implememented for compatibility with scikit-learn.
                Defaults to False.
            return_val (str):
                Specifies which prediction values to return.
                It can be "y", "s", "ei", or "all".

        Returns:
            np.ndarray:
                Predicted values of shape (n_samples,).
            np.ndarray:
                If self.return_std is True, returns the standard deviations of the predictions
                of shape (n_samples,).

        Examples:
            >>> import numpy as np
            >>> from spotpython.surrogate.kriging import Kriging
            >>> # Training data
            >>> X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            >>> y_train = np.array([0.1, 0.2, 0.3])
            >>> # Fit the Kriging model
            >>> model = Kriging().fit(X_train, y_train)
            >>> # Test data
            >>> X_test = np.array([[0.25, 0.25], [0.75, 0.75]])
            >>> # Predict responses
            >>> y_pred, sd, ei = model.predict(X_test)
            >>> print("Predictions:", y_pred)
        """
        self.return_std = return_std
        X = np.atleast_2d(X)
        if return_std:
            # Return predictions and standard deviations
            # Compatibility with scikit-learn
            self.return_std = True
            predictions, std_devs = zip(*[self._pred(x_i)[:2] for x_i in X])
            return np.array(predictions), np.array(std_devs)
        if return_val == "s":
            # Return only standard deviations
            self.return_std = True
            predictions, std_devs = zip(*[self._pred(x_i)[:2] for x_i in X])
            return np.array(std_devs)
        elif return_val == "all":
            # Return predictions, standard deviations, and expected improvements
            self.return_std = True
            self.return_ei = True
            predictions, std_devs, eis = zip(*[self._pred(x_i) for x_i in X])
            return np.array(predictions), np.array(std_devs), np.array(eis)
        elif return_val == "ei":
            # Return only neg. expected improvements
            self.return_ei = True
            predictions, eis = zip(*[(self._pred(x_i)[0], self._pred(x_i)[2]) for x_i in X])
            return np.array(eis)
        else:
            # Return only predictions (case "y")
            predictions = [self._pred(x_i)[0] for x_i in X]
            return np.array(predictions)

    def _kernel(self, X: np.ndarray, theta: np.ndarray, p: float) -> np.ndarray:
        """
        Computes the correlation matrix Psi using vectorized operations.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            theta (np.ndarray): Theta parameters of shape (n_features,).
            p (float): Power exponent.

        Returns:
            np.ndarray: The upper triangle of the correlation matrix Psi.
        """
        n_samples, n_features = X.shape
        Psi = np.zeros((n_samples, n_samples), dtype=float)
        # Calculate all pairwise differences:
        # X_expanded_rows will have shape (n_samples, 1, n_features)
        # X_expanded_cols will have shape (1, n_samples, n_features)
        # diff will have shape (n_samples, n_samples, n_features)
        diff = np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** p
        # Apply theta and sum over features
        # dist_matrix will have shape (n_samples, n_samples)
        dist_matrix = np.sum(theta * diff, axis=2)
        # Compute Psi using the exponential kernel
        Psi = np.exp(-dist_matrix)
        # Return only the upper triangle, as the matrix is symmetric
        # and the diagonal will be handled later.
        return np.triu(Psi, k=1)

    def likelihood(self, x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Computes the negative of the concentrated log-likelihood for a given set
        of log(theta) parameters using a power exponent p=1.99. Returns the
        negative log-likelihood, the correlation matrix Psi, and its Cholesky factor U.

        Args:
            x (np.ndarray):
                1D array of log(theta) parameters of length k. If self.method is "regression" or
                "reinterpolation", length is k+1 and the last element of x is the log(noise) parameter.

        Returns:
            (float, np.ndarray, np.ndarray):
                (negLnLike, Psi, U) where:
                - negLnLike (float): The negative concentrated log-likelihood.
                - Psi (np.ndarray): The correlation matrix.
                - U (np.ndarray): The Cholesky factor (or None if ill-conditioned).
        """
        # Extract data
        X = self.X_
        y = self.y_.flatten()

        if (self.method == "regression") or (self.method == "reinterpolation"):
            # case noise is True
            theta = x[:-1]
            # theta is in log scale, so transform it back:
            theta = 10.0**theta
            lambda_ = x[-1]
            # lambda is in log scale, so transform it back:
            lambda_ = 10.0**lambda_
        elif self.method == "interpolation":
            theta = x
            theta = 10.0**theta
            # use the original, untransformed eps:
            lambda_ = self.eps
        else:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation'")

        p = 1.99
        n = X.shape[0]
        one = np.ones(n)

        # Build correlation matrix
        Psi_upper_triangle = self._kernel(X, theta, p)
        Psi = Psi_upper_triangle + Psi_upper_triangle.T + np.eye(n) + np.eye(n) * lambda_

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

    def _pred(self, x: np.ndarray) -> float:
        """
        Computes a single-point Kriging prediction using the correlation matrix
        information. Internal helper method.

        Args:
            x (np.ndarray): 1D array of length k for the point at which to predict.

        Returns:
            float: The Kriging prediction at x.
            float: The standard deviation of the prediction.
            float: The NEGATIVE expected improvement at x.
        """
        X = self.X_
        y = self.y_.flatten()

        if self.method == "interpolation":
            theta = self.logtheta_lambda_
            theta = 10.0**theta
            # lambda is not transformed back:
            lambda_ = self.eps
        else:
            theta = self.logtheta_lambda_[:-1]
            theta = 10.0**theta
            lambda_ = self.logtheta_lambda_[-1]
            lambda_ = 10.0**lambda_

        U = self.U_

        p = 1.99
        n = X.shape[0]
        one = np.ones(n)

        # Compute mu
        y_tilde = np.linalg.solve(U, y)
        y_tilde = np.linalg.solve(U.T, y_tilde)
        one_tilde = np.linalg.solve(U, one)
        one_tilde = np.linalg.solve(U.T, one_tilde)
        mu = (one @ y_tilde) / (one @ one_tilde)

        resid = y - one * mu
        resid_tilde = np.linalg.solve(U, resid)
        resid_tilde = np.linalg.solve(U.T, resid_tilde)

        # Build psi
        psi = np.ones(n)
        for i in range(n):
            dist_vec = np.abs(X[i, :] - x) ** p
            psi[i] = np.exp(-np.sum(theta * dist_vec))

        # Compute SigmaSqr and SSqr
        if (self.method == "interpolation") or (self.method == "regression"):
            SigmaSqr = (resid @ resid_tilde) / n
            # Compute SSqr
            psi_tilde = np.linalg.solve(U, psi)
            psi_tilde = np.linalg.solve(U.T, psi_tilde)
            # Eq. (3.1) in [forr08a] without lambda:
            SSqr = SigmaSqr * (1 + lambda_ - psi @ psi_tilde)
        else:
            # method is "reinterpolation"
            Psi_adjusted = self.Psi_ - np.eye(n) * lambda_ + np.eye(n) * self.eps
            SigmaSqr = (resid @ np.linalg.solve(U.T, np.linalg.solve(U, Psi_adjusted @ resid_tilde))) / n
            # Compute Uint (Cholesky factor of the adjusted Psi matrix)
            Uint = np.linalg.cholesky(Psi_adjusted)

            # Compute SSqr
            psi_tilde = np.linalg.solve(Uint, psi)
            psi_tilde = np.linalg.solve(Uint.T, psi_tilde)
            SSqr = SigmaSqr * (1 - psi @ psi_tilde)

        # Compute s
        s = np.abs(SSqr) ** 0.5

        # Final prediction
        f = mu + psi @ resid_tilde

        # Compute ExpImp
        if self.return_ei:
            yBest = np.min(y)
            EITermOne = (yBest - f) * (0.5 + 0.5 * erf((1 / np.sqrt(2)) * ((yBest - f) / s)))
            EITermTwo = s * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((yBest - f) ** 2 / SSqr))
            ExpImp = np.log10(EITermOne + EITermTwo + self.eps)
            return float(f), float(s), float(-ExpImp)
        else:
            return float(f), float(s)

    def max_likelihood(self, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        """
        Maximizes the Kriging likelihood function using differential evolution
        over the range of log(theta) specified by bounds.

        Args:
            bounds (List[Tuple[float, float]]): Sequence of (low, high) bounds for log(theta).

        Returns:
            (np.ndarray, float): (best_x, best_fun) where best_x is the
            optimal log(theta) array and best_fun is the minimized negative log-likelihood.
        """

        def objective(logtheta_lambda):
            neg_ln_like, _, _ = self.likelihood(logtheta_lambda)
            return neg_ln_like

        result = differential_evolution(objective, bounds)
        return result.x, result.fun

    def plot(self, show: Optional[bool] = True) -> None:
        """
        This function plots 1D and 2D surrogates.
        Only for compatibility with the old Kriging implementation.

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
            x = linspace(self.min_X[0], self.max_X[0], num=n_grid)
            y = self.predict(x)
            plt.figure()
            plt.plot(x, y, "k")
            if show:
                plt.show()

        if self.k == 2:
            fig = pylab.figure(figsize=(9, 6))
            n_grid = 100
            x = linspace(self.min_X[0], self.max_X[0], num=n_grid)
            y = linspace(self.min_X[1], self.max_X[1], num=n_grid)
            X, Y = meshgrid(x, y)
            # Predict based on the optimized results
            zz = array([self.predict(array([x, y]), return_val="all") for x, y in zip(ravel(X), ravel(Y))])
            zs = zz[:, 0, :]
            zse = zz[:, 1, :]
            Z = zs.reshape(X.shape)
            Ze = zse.reshape(X.shape)

            nat_point_X = self.X_[:, 0]
            nat_point_Y = self.X_[:, 1]
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


# Additional functions for plotting the Kriging surrogate model
# ------------------------------------------------------------


def plot1d(model, X: np.ndarray, y: np.ndarray, show: Optional[bool] = True) -> None:
    """
    Plots the 1D Kriging surrogate model.

    Args:
        model (object): A fitted Kriging model.
        X (np.ndarray): Training input data of shape (n_samples, 1).
        y (np.ndarray): Training target values of shape (n_samples,).
        show (bool): If True, displays the plot. Defaults to True.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from spotpython.surrogate.kriging import Kriging
        >>> # Training data
        >>> X_train = np.array([[0.0], [0.5], [1.0]])
        >>> y_train = np.array([0.1, 0.2, 0.3])
        >>> # Initialize and fit the Kriging model
        >>> model = Kriging().fit(X_train, y_train)
        >>> # Plot the 1D Kriging surrogate
        >>> plot1d(model, X_train, y_train)
    """
    if X.shape[1] != 1:
        raise ValueError("plot1d is only supported for 1D input data.")

    _ = plt.figure(figsize=(9, 6))
    n_grid = 100
    x = linspace(X[:, 0].min(), X[:, 0].max(), num=n_grid).reshape(-1, 1)
    y_pred, y_std = model.predict(x, return_std=True)

    plt.plot(x, y_pred, "k", label="Prediction")
    plt.fill_between(
        x.ravel(),
        y_pred - 1.96 * y_std,
        y_pred + 1.96 * y_std,
        alpha=0.2,
        label="95% Confidence Interval",
    )
    plt.scatter(X, y, color="red", label="Training Data")
    plt.xlabel("X")
    plt.ylabel("Prediction")
    plt.title("1D Kriging Surrogate")
    plt.legend()
    if show:
        plt.show()


def plot2d(model, X: np.ndarray, y: np.ndarray, show: Optional[bool] = True, alpha=0.8) -> None:
    """
    Plots the 2D Kriging surrogate model.

    Args:
        model (object): A fitted Kriging model.
        X (np.ndarray): Training input data of shape (n_samples, 2).
        y (np.ndarray): Training target values of shape (n_samples,).
        show (bool): If True, displays the plot. Defaults to True.
        alpha (float): Transparency level for 3D surface plots. Defaults to 0.8.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from spotpython.surrogate.kriging import Kriging
        >>> # Training data
        >>> X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        >>> y_train = np.array([0.1, 0.2, 0.3])
        >>> # Initialize and fit the Kriging model
        >>> model = Kriging().fit(X_train, y_train)
        >>> # Plot the 2D Kriging surrogate
        >>> plot2d(model, X_train, y_train)
    """
    if X.shape[1] != 2:
        raise ValueError("plot2d is only supported for 2D input data.")

    fig = plt.figure(figsize=(12, 10))
    n_grid = 100
    x1 = linspace(X[:, 0].min(), X[:, 0].max(), num=n_grid)
    x2 = linspace(X[:, 1].min(), X[:, 1].max(), num=n_grid)
    X1, X2 = meshgrid(x1, x2)
    grid_points = array([X1.ravel(), X2.ravel()]).T

    y_pred, y_std = model.predict(grid_points, return_std=True)
    Z_pred = y_pred.reshape(X1.shape)
    Z_std = y_std.reshape(X1.shape)

    # Plot predicted values
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.plot_surface(X1, X2, Z_pred, cmap="viridis", alpha=alpha)
    ax1.set_title("Prediction Surface")
    ax1.set_xlabel("X1")
    ax1.set_ylabel("X2")
    ax1.set_zlabel("Prediction")

    # Plot prediction error
    ax2 = fig.add_subplot(222, projection="3d")
    ax2.plot_surface(X1, X2, Z_std, cmap="viridis", alpha=alpha)
    ax2.set_title("Prediction Error Surface")
    ax2.set_xlabel("X1")
    ax2.set_ylabel("X2")
    ax2.set_zlabel("Error")

    # Contour plot of predicted values
    ax3 = fig.add_subplot(223)
    contour = ax3.contourf(X1, X2, Z_pred, cmap="viridis", levels=30)
    plt.colorbar(contour, ax=ax3)
    ax3.scatter(X[:, 0], X[:, 1], color="red", label="Training Data")
    ax3.set_title("Prediction Contour")
    ax3.set_xlabel("X1")
    ax3.set_ylabel("X2")
    ax3.legend()

    # Contour plot of prediction error
    ax4 = fig.add_subplot(224)
    contour = ax4.contourf(X1, X2, Z_std, cmap="viridis", levels=30)
    plt.colorbar(contour, ax=ax4)
    ax4.scatter(X[:, 0], X[:, 1], color="red", label="Training Data")
    ax4.set_title("Error Contour")
    ax4.set_xlabel("X1")
    ax4.set_ylabel("X2")
    ax4.legend()

    if show:
        plt.show()


def plotkd(
    model,
    X: np.ndarray,
    y: np.ndarray,
    i: int,
    j: int,
    show: Optional[bool] = True,
    alpha=0.8,
    eps=1e-3,
    var_names: Optional[List[str]] = None,
) -> None:
    """
    Plots the Kriging surrogate model for k-dimensional input data by varying two dimensions (i, j).

    Args:
        model (object): A fitted Kriging model.
        X (np.ndarray): Training input data of shape (n_samples, k).
        y (np.ndarray): Training target values of shape (n_samples,).
        i (int): The first dimension to vary.
        j (int): The second dimension to vary.
        show (bool): If True, displays the plot. Defaults to True.
        alpha (float): Transparency level for 3D surface plots. Defaults to 0.8.
        eps (float): Tolerance for considering points as "on the surface". Defaults to 1e-3.
        var_names (List[str], optional): A list of three strings for axis labels.
            The first entry is for the x-axis, the second for the y-axis, and the third for the z-axis.
            If empty or None, default axis labels are used.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from spotpython.surrogate.kriging import Kriging, plotkd
        >>> # Training data
        >>> X_train = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
        >>> y_train = np.array([0.1, 0.2, 0.3])
        >>> # Initialize and fit the Kriging model
        >>> model = Kriging().fit(X_train, y_train)
        >>> # Plot the 3D Kriging surrogate
        >>> plotkd(model, X_train, y_train, 0, 1)
    """
    k = X.shape[1]
    if i >= k or j >= k:
        raise ValueError(f"Dimensions i and j must be less than the number of features (k={k}).")
    if i == j:
        raise ValueError("Dimensions i and j must be different.")

    # Compute the mean values for all dimensions
    mean_values = X.mean(axis=0)

    # Create a grid for the two varied dimensions
    n_grid = 100
    x_i = linspace(X[:, i].min(), X[:, i].max(), num=n_grid)
    x_j = linspace(X[:, j].min(), X[:, j].max(), num=n_grid)
    X_i, X_j = meshgrid(x_i, x_j)

    # Prepare the grid points for prediction
    grid_points = np.zeros((X_i.size, k))
    grid_points[:, i] = X_i.ravel()
    grid_points[:, j] = X_j.ravel()

    # Set the remaining dimensions to their mean values
    for dim in range(k):
        if dim != i and dim != j:
            grid_points[:, dim] = mean_values[dim]

    # Predict the values and standard deviations
    y_pred, y_std = model.predict(grid_points, return_std=True)
    Z_pred = y_pred.reshape(X_i.shape)
    Z_std = y_std.reshape(X_i.shape)

    # Plot the results
    fig = plt.figure(figsize=(12, 10))

    # Plot predicted values
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.plot_surface(X_i, X_j, Z_pred, cmap="viridis", alpha=alpha)
    ax1.set_title("Prediction Surface")
    ax1.set_xlabel(var_names[0] if var_names else f"Dimension {i}")
    ax1.set_ylabel(var_names[1] if var_names else f"Dimension {j}")
    ax1.set_zlabel(var_names[2] if var_names else "Prediction")

    # Add input points to the prediction surface
    for idx in range(X.shape[0]):
        x_point = X[idx, i]
        y_point = X[idx, j]
        z_actual = y[idx]
        z_predicted = model.predict(X[idx].reshape(1, -1))[0]

        if z_actual > z_predicted + eps:
            color = "red"
        elif z_actual < z_predicted - eps:
            color = "green"
        else:
            color = "white"

        ax1.scatter(x_point, y_point, z_actual, color=color, s=50, edgecolor="black")

    # Plot prediction error
    ax2 = fig.add_subplot(222, projection="3d")
    ax2.plot_surface(X_i, X_j, Z_std, cmap="viridis", alpha=alpha)
    ax2.set_title("Prediction Error Surface")
    ax2.set_xlabel(var_names[0] if var_names else f"Dimension {i}")
    ax2.set_ylabel(var_names[1] if var_names else f"Dimension {j}")
    ax2.set_zlabel(var_names[2] if var_names else "Error")

    # Add input points to the error surface
    for idx in range(X.shape[0]):
        x_point = X[idx, i]
        y_point = X[idx, j]
        z_actual = y[idx]
        z_predicted = model.predict(X[idx].reshape(1, -1))[0]

        if z_actual > z_predicted + eps:
            color = "red"
        elif z_actual < z_predicted - eps:
            color = "green"
        else:
            color = "white"

        ax2.scatter(x_point, y_point, abs(z_actual - z_predicted), color=color, s=50, edgecolor="black")

    # Contour plot of predicted values
    ax3 = fig.add_subplot(223)
    contour = ax3.contourf(X_i, X_j, Z_pred, cmap="viridis", levels=30)
    plt.colorbar(contour, ax=ax3)
    for idx in range(X.shape[0]):
        x_point = X[idx, i]
        y_point = X[idx, j]
        z_actual = y[idx]
        z_predicted = model.predict(X[idx].reshape(1, -1))[0]

        if z_actual > z_predicted + eps:
            color = "red"
        elif z_actual < z_predicted - eps:
            color = "green"
        else:
            color = "white"

        ax3.scatter(x_point, y_point, color=color, s=50, edgecolor="black")
    ax3.set_title("Prediction Contour")
    ax3.set_xlabel(var_names[0] if var_names else f"Dimension {i}")
    ax3.set_ylabel(var_names[1] if var_names else f"Dimension {j}")

    # Contour plot of prediction error
    ax4 = fig.add_subplot(224)
    contour = ax4.contourf(X_i, X_j, Z_std, cmap="viridis", levels=30)
    plt.colorbar(contour, ax=ax4)
    for idx in range(X.shape[0]):
        x_point = X[idx, i]
        y_point = X[idx, j]
        z_actual = y[idx]
        z_predicted = model.predict(X[idx].reshape(1, -1))[0]

        if z_actual > z_predicted + eps:
            color = "red"
        elif z_actual < z_predicted - eps:
            color = "green"
        else:
            color = "white"

        ax4.scatter(x_point, y_point, color=color, s=50, edgecolor="black")
    ax4.set_title("Error Contour")
    ax4.set_xlabel(var_names[0] if var_names else f"Dimension {i}")
    ax4.set_ylabel(var_names[1] if var_names else f"Dimension {j}")

    if show:
        plt.show()
