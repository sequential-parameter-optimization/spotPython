import numpy as np
from numpy.linalg import LinAlgError, cond, svd
from numpy.random import RandomState
from typing import Dict, Tuple, List, Optional
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.special import erf
import matplotlib.pyplot as plt
from numpy import linspace, append
from scipy.spatial.distance import cdist, pdist, squareform
from spotpython.surrogate.plot import plotkd


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
        negLnLike (float): The negative log-likelihood of the model.
        Psi_ (np.ndarray): The correlation matrix after fit().
        method (str): The fitting method used, can be "interpolation", "regression", or "reinterpolation".
        isotropic (bool): Whether the model is isotropic or not.
        approximation (str or None): Approximation method used, e.g., "nystroem" or None. Defaults to None.
        n_components_nystroem (int or None): Number of components for Nyström approximation. Defaults to None.

    Methods:
        __init__: Initializes the Kriging model with hyperparameters.
        _get_eps: Returns the square root of machine epsilon.
        _set_variable_types: Sets variable types for the model.
        get_model_params: Returns additional model parameters not included in get_params().
        _update_log: Updates the log with current model parameters.
        fit: Fits the Kriging model to training data X and y.
        predict: Predicts the Kriging response at a set of points X.
        build_Psi: Constructs a new correlation matrix Psi.
        likelihood: Computes the negative concentrated log-likelihood and correlation matrix.
        build_psi_vec: Builds the psi vector for predictive methods.
        _pred: Computes a single-point Kriging prediction.
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
        metric_factorial="canberra",
        isotropic: bool = False,
        approximation: Optional[str] = None,  # Default is None
        n_components_nystroem: Optional[int] = None,  # Number of components for Nyström
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
            isotropic (bool, optional):
                If True, the model is isotropic, meaning all variables are treated equally (only one theta value is used).
                If False, the model can handle different theta values, one for each dimension. Defaults to False.
        """
        if eps is None:
            self.eps = self._get_eps()
        else:
            if eps <= 0:
                raise ValueError("eps must be positive")
            self.eps = eps
        self.penalty = penalty
        self.noise = noise
        self.var_type = var_type  # Store original variable types from input
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
        self.n_theta = None  # Will be set in fit() based on self.k_orig
        self.isotropic = isotropic
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

        self.log = {}
        self.log["negLnLike"] = []
        self.log["theta"] = []
        self.log["p"] = []
        self.log["Lambda"] = []

        self.logtheta_lambda_ = None
        self.U_ = None
        self.X_ = None  # Training data for the Kriging model (could be Nyström features)
        self.y_ = None
        self.negLnLike = None
        self.Psi_ = None
        if method not in ["interpolation", "regression", "reinterpolation"]:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation']")
        self.method = method
        self.return_ei = False
        self.return_std = False

        # Nyström specific initializations
        if approximation is not None and approximation.lower() not in ["nystroem"]:
            raise ValueError("approximation must be 'nystroem' or None")
        self.approximation = approximation.lower() if approximation is not None else None
        self.n_components_nystroem = n_components_nystroem
        self._nystroem_X_subset = None  # Stores the actual subset of original X points
        self._nystroem_components_sqrt_inv = None  # Stores U_m @ Sigma_m^{-1/2} from SVD of K_mm
        self._nystroem_n_components_actual = None  # Actual components used after filtering small eigenvalues
        self.rng = RandomState(seed)  # Random number generator for reproducible sampling

        # Attributes to store original input dimensions and masks [7]
        self.k_orig = None  # Original number of features
        self.ordered_mask_orig = None
        self.factor_mask_orig = None

        # Attributes for the Kriging model's feature space (could be Nyström transformed)
        self.k_orig = None  # Current number of features for the Kriging model
        self.var_type_kriging = None  # Variable types for the Kriging model's feature space
        self.ordered_mask_kriging = None
        self.factor_mask_kriging = None

    def _get_eps(self) -> float:
        """
        Returns the square root of the machine epsilon.
        """
        eps = np.finfo(float).eps
        return np.sqrt(eps)

    def _set_variable_types(self, k_original: int) -> None:
        """
        Set variable types and masks for the ORIGINAL input data.
        These are used when computing kernels for Nyström transformation. [7]
        """
        self.k_orig = k_original
        if len(self.var_type) < k_original:
            self.var_type = ["num"] * k_original
        var_type_array_orig = np.array(self.var_type)
        self.ordered_mask_orig = np.isin(var_type_array_orig, ["int", "num", "float"])
        self.factor_mask_orig = var_type_array_orig == "factor"

    def _set_kriging_model_feature_types(self, k_current: int) -> None:
        """
        Set variable types and masks for the Kriging model's current feature space.
        These are used when the Kriging model operates on its (possibly transformed) X_.
        """
        self.k_orig = k_current
        # If Nyström is used, features are always numerical
        if self.approximation == "nystroem":
            self.var_type_kriging = ["num"] * k_current
            self.ordered_mask_kriging = np.array([True] * k_current)
            self.factor_mask_kriging = np.array([False] * k_current)
        else:
            # For standard Kriging, use original masks and types
            self.var_type_kriging = self.var_type
            self.ordered_mask_kriging = self.ordered_mask_orig
            self.factor_mask_kriging = self.factor_mask_orig

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
            self.log["p"] = append(self.log["p"], self.p_val)
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
                p = self.p_val.copy()
                self.spot_writer.add_scalars("spot_p", {f"p_{i}": p[i] for i in range(self.n_p)}, self.counter + self.log_length)
            self.spot_writer.flush()

    def _compute_kernel_matrix_for_original_features(self, X1: np.ndarray, X2: np.ndarray, theta10_for_kernel: np.ndarray) -> np.ndarray:
        """
        Computes the kernel (correlation) matrix between two sets of ORIGINAL features X1 and X2,
        respecting original variable types (ordered/factor).
        This helper is specifically for the Nyström transformation stage.
        """
        # TODO: Check if n1, n2 are correct:
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        D = np.zeros((n1, n2), dtype=np.float64)

        if self.ordered_mask_orig.any():
            X1_ordered = X1[:, self.ordered_mask_orig]
            X2_ordered = X2[:, self.ordered_mask_orig]
            # Ensure theta10_for_kernel has the correct length for ordered variables
            theta10_ordered = theta10_for_kernel[self.ordered_mask_orig] if theta10_for_kernel.size > 1 else theta10_for_kernel * np.ones(X1_ordered.shape[1])
            D_ordered = cdist(X1_ordered, X2_ordered, metric="sqeuclidean", w=theta10_ordered)
            D += D_ordered

        if self.factor_mask_orig.any():
            X1_factor = X1[:, self.factor_mask_orig]
            X2_factor = X2[:, self.factor_mask_orig]
            # Ensure theta10_for_kernel has the correct length for factor variables
            theta10_factor = theta10_for_kernel[self.factor_mask_orig] if theta10_for_kernel.size > 1 else theta10_for_kernel * np.ones(X1_factor.shape[1])
            D_factor = cdist(X1_factor, X2_factor, metric=self.metric_factorial, w=theta10_factor)
            D += D_factor

        return np.exp(-D)

    def fit(self, X: np.ndarray, y: np.ndarray, bounds: Optional[List[Tuple[float, float]]] = None) -> "Kriging":
        """
        Fits the Kriging model to training data X and y. This method is compatible with scikit-learn and
        uses differential evolution to optimize the hyperparameters (log(theta)).
        If 'approximation' is set to 'nystroem', the input data X is first transformed using Nyström approximation.

        Args:
            X (np.ndarray):
                Training input data of shape (n_samples, n_features).
            y (np.ndarray):
                Target values of shape (n_samples,) or (n_samples, 1).
            bounds (Optional[List[Tuple[float, float]]]):
                Bounds for each dimension of log(theta). If None, defaults to [(-3, 2)] * n_features.

        Returns:
            Kriging:
                The fitted Kriging model instance (self).
        """
        X_orig = np.asarray(X)
        y_orig = np.asarray(y).flatten()

        self.n_orig, k_orig_val = X_orig.shape  # Store original dimensions
        self._set_variable_types(k_orig_val)  # Set masks for original feature space

        # Nyström Approximation Preprocessing
        if self.approximation == "nystroem":
            if self.n_components_nystroem is None:
                self.n_components_nystroem = min(self.n_orig, 100)  # Default to 100 or n_orig if smaller
            if not (1 <= self.n_components_nystroem <= self.n_orig):
                raise ValueError(f"n_components_nystroem must be between 1 and n_samples ({self.n_orig}), got {self.n_components_nystroem}")

            # Sample subset for Nyström
            self._nystroem_X_subset_indices = self.rng.choice(self.n_orig, size=self.n_components_nystroem, replace=False)
            self._nystroem_X_subset = X_orig[self._nystroem_X_subset_indices]

            # Use initial theta (e.g., 10^0 = 1 for all original dimensions) for Nyström kernel computation
            # The actual Kriging theta will be optimized later for the Nyström features.
            initial_theta_for_nystroem = np.power(10.0, np.zeros(self.k_orig))

            # Compute K_mm (kernel matrix of sampled points)
            K_mm = self._compute_kernel_matrix_for_original_features(self._nystroem_X_subset, self._nystroem_X_subset, initial_theta_for_nystroem)

            # Regularization for K_mm to ensure numerical stability before SVD
            K_mm += self._get_eps() * np.eye(K_mm.shape)

            # Perform SVD on K_mm
            U_mm, s_mm, _ = svd(K_mm)

            # Filter small eigenvalues and determine actual components
            valid_s = s_mm > self._get_eps()  # Threshold for eigenvalues
            self._nystroem_n_components_actual = np.sum(valid_s)

            if self._nystroem_n_components_actual == 0:
                print("Warning: All Nyström eigenvalues are effectively zero. Falling back to standard Kriging.")
                # If no valid components, fall back to standard Kriging
                self.approximation = None
                self.X_ = X_orig
                self.y_ = y_orig
                self._set_kriging_model_feature_types(self.k_orig)  # Set Kriging masks for original features
            else:
                # Compute components_sqrt_inv = U_mm @ diag(1 / sqrt(s_mm))
                self._nystroem_components_sqrt_inv = U_mm[:, valid_s] @ np.diag(1.0 / np.sqrt(s_mm[valid_s]))

                # Transform original X to Nyström features
                K_nm = self._compute_kernel_matrix_for_original_features(X_orig, self._nystroem_X_subset, initial_theta_for_nystroem)
                X_nystroem = K_nm @ self._nystroem_components_sqrt_inv

                self.X_ = X_nystroem
                self.y_ = y_orig  # y remains the same
                self._set_kriging_model_feature_types(self.X_.shape[1])  # Set Kriging masks for Nyström features
                # print(f"Nyström approximation applied. Original dimensions: {self.k_orig}, Nyström features: {self.k_orig}")

        else:  # Standard Kriging (approximation is None)
            self.X_ = X_orig
            self.y_ = y_orig
            self._set_kriging_model_feature_types(self.k_orig)  # Set Kriging masks for original features
            # print(f"Standard Kriging. Dimensions: {self.k_orig}")

        self.n = self.X_.shape[0]  # Update n for the (possibly transformed) X_

        # Kriging fitting part (operates on self.X_ which might be Nyström features)
        if self.isotropic:
            self.n_theta = 1
            # print(f"Isotropic model: n_theta set to {self.n_theta}")
        else:
            self.n_theta = self.k_orig  # n_theta based on current Kriging feature dimension
            # print(f"Anisotropic model: n_theta set to {self.n_theta}")

        self.min_X = np.min(self.X_, axis=0)
        self.max_X = np.max(self.X_, axis=0)

        if bounds is None:
            if self.method == "interpolation":
                bounds = [(self.min_theta, self.max_theta)] * self.k_orig
            else:
                bounds = [(self.min_theta, self.max_theta)] * self.k_orig + [(self.min_Lambda, self.max_Lambda)]

        if self.optim_p:
            n_p_to_optimize = self.n_p if self.n_p == 1 else self.k_orig  # Number of p values to optimize (1 or k)
            bounds += [(self.min_p, self.max_p)] * n_p_to_optimize

        self.logtheta_lambda_, _ = self.max_likelihood(bounds)

        # store theta and Lambda in log scale
        if (self.method == "regression") or (self.method == "reinterpolation"):
            # select the first n_theta values from logtheta_lambda_:
            self.theta = self.logtheta_lambda_[: self.n_theta]
            self.Lambda = self.logtheta_lambda_[self.n_theta : self.n_theta + 1]
            if self.optim_p:
                self.p_val = self.logtheta_lambda_[self.n_theta + 1 : self.n_theta + 1 + self.n_p]
        else:
            self.theta = self.logtheta_lambda_[: self.n_theta]
            self.Lambda = None
            if self.optim_p:
                self.p_val = self.logtheta_lambda_[self.n_theta : self.n_theta + self.n_p]

        # Once logtheta_lambda is found, compute the final correlation matrix
        self.negLnLike, self.Psi_, self.U_ = self.likelihood(self.logtheta_lambda_)

        # Update log with the current values
        self._update_log()
        return self

    def predict(self, X: np.ndarray, return_std=False, return_val: str = "y") -> np.ndarray:
        """
        Predicts the Kriging response at a set of points X. This method is compatible with scikit-learn.
        If 'approximation' is set to 'nystroem', the input data X is first transformed using
        the Nyström components learned during fitting.

        Args:
            X (np.ndarray):
                Array of shape (n_samples, n_features) containing the points at which
                to predict the Kriging response.
            return_std (bool, optional):
                If True, returns the standard deviation of the predictions as well.
                Implemented for compatibility with scikit-learn.
                Defaults to False.
            return_val (str):
                Specifies which prediction values to return.
                It can be "y", "s", "ei", or "all".
                Defaults to "y".

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
        X_pred_orig = np.atleast_2d(X)

        if self.approximation == "nystroem":
            if self._nystroem_components_sqrt_inv is None or self._nystroem_X_subset is None:
                raise RuntimeError("Kriging model with Nyström approximation is not fitted yet. Call fit() first.")

            # Use the same initial theta for Nyström kernel computation as in fit()
            initial_theta_for_nystroem = np.power(10.0, np.zeros(self.k_orig))

            # Transform prediction points using the fitted Nyström components
            K_test_subset = self._compute_kernel_matrix_for_original_features(X_pred_orig, self._nystroem_X_subset, initial_theta_for_nystroem)
            X_pred_for_kriging = K_test_subset @ self._nystroem_components_sqrt_inv
        else:
            X_pred_for_kriging = X_pred_orig

        self.return_std = return_std

        # Original predict logic using X_pred_for_kriging
        if return_std:
            # Return predictions and standard deviations
            # Compatibility with scikit-learn
            self.return_std = True
            predictions, std_devs = zip(*[self._pred(x_i)[:2] for x_i in X_pred_for_kriging])
            return np.array(predictions), np.array(std_devs)
        if return_val == "s":
            # Return only standard deviations
            self.return_std = True
            predictions, std_devs = zip(*[self._pred(x_i)[:2] for x_i in X_pred_for_kriging])
            return np.array(std_devs)
        elif return_val == "all":
            # Return predictions, standard deviations, and expected improvements
            self.return_std = True
            self.return_ei = True
            predictions, std_devs, eis = zip(*[self._pred(x_i) for x_i in X_pred_for_kriging])
            return np.array(predictions), np.array(std_devs), np.array(eis)
        elif return_val == "ei":
            # Return only neg. expected improvements
            self.return_ei = True
            predictions, eis = zip(*[(self._pred(x_i)[0], self._pred(x_i)[2]) for x_i in X_pred_for_kriging])
            return np.array(eis)
        else:
            # Return only predictions (case "y")
            predictions = [self._pred(x_i)[0] for x_i in X_pred_for_kriging]
            return np.array(predictions)

    def build_Psi(self) -> np.ndarray:
        """
        Constructs a new (n x n) correlation matrix Psi to reflect new data or a change in hyperparameters.
        Operates on the Kriging model's current feature space (self.X_).
        This method uses `theta`, `p`, and coded `X` values to construct the
        correlation matrix as described in [Forr08a, p.57].

        Returns:
            np.ndarray:
                Upper triangular part of the correlation matrix Psi (excluding diagonal).

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
            n, k_current = self.X_.shape
            theta10 = np.power(10.0, self.theta)

            if self.n_theta == 1:
                theta10 = theta10 * np.ones(k_current)  # Ensure theta10 matches current feature dimension

            Psi = np.zeros((n, n), dtype=np.float64)

            # Use Kriging model's own masks (self.ordered_mask_kriging, self.factor_mask_kriging)
            # These masks are set in _set_kriging_model_feature_types based on whether Nyström is active.
            if self.ordered_mask_kriging.any():
                X_ordered_for_kriging = self.X_[:, self.ordered_mask_kriging]
                theta10_ordered_for_kriging = theta10[self.ordered_mask_kriging]
                D_ordered = squareform(pdist(X_ordered_for_kriging, metric="sqeuclidean", w=theta10_ordered_for_kriging))
                Psi += D_ordered

            # Add the contribution of factor variables to the distance matrix
            if self.factor_mask_kriging.any():
                X_factor_for_kriging = self.X_[:, self.factor_mask_kriging]
                theta10_factor_for_kriging = theta10[self.factor_mask_kriging]
                D_factor = squareform(pdist(X_factor_for_kriging, metric=self.metric_factorial, w=theta10_factor_for_kriging))
                Psi += D_factor

            # Calculate correlation from distance
            Psi = np.exp(-Psi)
            # Check for infinite values
            self.inf_Psi = np.isinf(Psi).any()
            # Calculate condition number
            self.cnd_Psi = cond(Psi)
            # Return only the upper triangle, as the matrix is symmetric
            # and the diagonal will be handled later.
            return np.triu(Psi, k=1)
        except LinAlgError as err:
            print(f"Building Psi failed. Error: {err}, Type: {type(err)}")
            raise

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
            # theta = x[:-1]
            self.theta = x[: self.n_theta]
            # lambda_ = x[-1]
            lambda_ = x[self.n_theta : self.n_theta + 1]
            # lambda is in log scale, so transform it back:
            lambda_ = 10.0**lambda_
            if self.optim_p:
                self.p_val = x[self.n_theta + 1 : self.n_theta + 1 + self.n_p]
        elif self.method == "interpolation":
            # theta = x
            self.theta = x[: self.n_theta]
            # use the original, untransformed eps:
            lambda_ = self.eps
            if self.optim_p:
                self.p_val = x[self.n_theta : self.n_theta + self.n_p]
        else:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation'")

        n = X.shape[0]
        one = np.ones(n)

        Psi_upper_triangle = self.build_Psi()

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

    def build_psi_vec(self, x: np.ndarray) -> np.ndarray:
        """
        Build the psi vector required for predictive methods.
        Operates on a single point 'x' in the Kriging model's current feature space.

        Args:
            x (np.ndarray): 1D array of length k for the point at which to compute psi.

        Returns:
            np.ndarray: The psi vector of shape (n,).

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
        try:
            n, k_current = self.X_.shape
            psi = np.zeros(n)
            theta10 = np.power(10.0, self.theta)

            if self.n_theta == 1:
                theta10 = theta10 * np.ones(k_current)

            D = np.zeros(n)

            # Use Kriging model's own masks (self.ordered_mask_kriging, self.factor_mask_kriging)
            # These masks are set in _set_kriging_model_feature_types based on whether Nyström is active.
            if self.ordered_mask_kriging.any():
                X_ordered_for_kriging = self.X_[:, self.ordered_mask_kriging]
                x_ordered_for_kriging = x[self.ordered_mask_kriging]
                theta10_ordered_for_kriging = theta10[self.ordered_mask_kriging]
                D += cdist(x_ordered_for_kriging.reshape(1, -1), X_ordered_for_kriging, metric="sqeuclidean", w=theta10_ordered_for_kriging).ravel()

            if self.factor_mask_kriging.any():
                X_factor_for_kriging = self.X_[:, self.factor_mask_kriging]
                x_factor_for_kriging = x[self.factor_mask_kriging]
                theta10_factor_for_kriging = theta10[self.factor_mask_kriging]
                D += cdist(x_factor_for_kriging.reshape(1, -1), X_factor_for_kriging, metric=self.metric_factorial, w=theta10_factor_for_kriging).ravel()

            psi = np.exp(-D)
            return psi
        except np.linalg.LinAlgError as err:
            print(f"Building psi failed due to a linear algebra error: {err}. Error type: {type(err)}")
            raise

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
        y = self.y_.flatten()

        if (self.method == "regression") or (self.method == "reinterpolation"):
            self.theta = self.logtheta_lambda_[: self.n_theta]
            lambda_ = self.logtheta_lambda_[self.n_theta : self.n_theta + 1]
            # lambda is in log scale, so transform it back:
            lambda_ = 10.0**lambda_
            if self.optim_p:
                self.p_val = self.logtheta_lambda_[self.n_theta + 1 : self.n_theta + 1 + self.n_p]
        elif self.method == "interpolation":
            self.theta = self.logtheta_lambda_[: self.n_theta]
            # use the original, untransformed eps:
            lambda_ = self.eps
            if self.optim_p:
                self.p_val = self.logtheta_lambda_[self.n_theta : self.n_theta + self.n_p]
        else:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation'")

        U = self.U_
        n = self.X_.shape[0]
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

        psi = self.build_psi_vec(x)

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

        SSqr = SSqr.item()
        # Compute s
        s = np.abs(SSqr) ** 0.5

        # Final prediction
        f = mu + psi @ resid_tilde
        # print(f"Prediction at {x}: f={f}, s={s}, SigmaSqr={SigmaSqr}, SSqr={SSqr}")

        # Compute ExpImp
        if self.return_ei:
            yBest = np.min(y)
            EITermOne = (yBest - f) * (0.5 + 0.5 * erf((1 / np.sqrt(2)) * ((yBest - f) / s)))
            EITermTwo = s * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((yBest - f) ** 2 / SSqr))
            ExpImp = np.log10(EITermOne + EITermTwo + self.eps)
            return f.item(), s.item(), (-ExpImp).item()
        else:
            return f.item(), s.item()

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

        # result = differential_evolution(objective, bounds, seed=self.seed, maxiter=self.model_fun_evals)
        # print(f"bounds for differential_evolution: {bounds}")
        result = differential_evolution(objective, bounds)
        return result.x, result.fun

    def plot(self, i: int = 0, j: int = 1, show: Optional[bool] = True, add_points: bool = True) -> None:
        """
        This function plots 1D and 2D surrogates.
        Only for compatibility with the old Kriging implementation.

        Args:
            self (object):
                The Kriging object.
            i (int):
                The index of the first variable to plot.
            j (int):
                The index of the second variable to plot.
            show (bool):
                If `True`, the plots are displayed.
                If `False`, `plt.show()` should be called outside this function.
            add_points (bool):
                If `True`, the points from the design are added to the plot.

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
        if self.k_orig == 1:
            n_grid = 100
            x = linspace(self.min_X[0], self.max_X[0], num=n_grid)
            y = self.predict(x)
            plt.figure()
            plt.plot(x, y, "k")
            if show:
                plt.show()
        else:
            plotkd(model=self, X=self.X_, y=self.y_, i=i, j=j, show=show, var_type=self.var_type, add_points=True)
