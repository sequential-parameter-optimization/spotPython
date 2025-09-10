import numpy as np
from numpy.linalg import LinAlgError, cond, eigh, svd
from typing import Dict, Tuple, List, Optional
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.special import erf
import matplotlib.pyplot as plt
from numpy import linspace, append
from scipy.spatial.distance import cdist, pdist, squareform
from spotpython.surrogate.plot import plotkd  # Assuming this import is valid in the spotpython environment

from numpy.random import RandomState


class Kriging_Nystroem(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible Kriging model class for regression tasks.
    Provides methods for likelihood evaluation, predictions, and hyperparameter optimization.
    """

    def __init__(
        self,
        eps: float = None,
        penalty: float = 1e4,
        method="regression",
        noise: bool = False,
        var_type: List[str] = ["num"],  # Describes original input variables
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
        # New Nyström approximation parameters
        approximation: Optional[str] = None,  # Default is None
        n_components_nystroem: Optional[int] = None,  # Number of components for Nyström
        **kwargs,
    ):
        if eps is None:
            self.eps = self._get_eps()
        else:
            if eps <= 0:
                raise ValueError("eps must be positive")
            self.eps = eps
        self.penalty = penalty
        self.noise = noise
        self.var_type_input = var_type  # Store original variable types from input [3]
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
        self.n_theta = None  # Will be set in fit() based on self.k [4]
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
        self.X_ = None  # Training data for the Kriging model (could be Nyström features) [5]
        self.y_ = None
        self.negLnLike = None
        self.Psi_ = None
        if method not in ["interpolation", "regression", "reinterpolation"]:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation']")
        self.method = method
        self.return_ei = False
        self.return_std = False

        # Nyström specific initializations [6]
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
        self.k = None  # Current number of features for the Kriging model
        self.var_type_kriging = None  # Variable types for the Kriging model's feature space
        self.ordered_mask_kriging = None
        self.factor_mask_kriging = None

    def _get_eps(self) -> float:
        """ Returns the square root of the machine epsilon. """[7]
        eps = np.finfo(float).eps
        return np.sqrt(eps)

    def _set_original_variable_types(self, k_original: int) -> None:
        """
        Set variable types and masks for the ORIGINAL input data.
        These are used when computing kernels for Nyström transformation. [7]
        """
        self.k_orig = k_original
        if len(self.var_type_input) < k_original:
            self.var_type_input = ["num"] * k_original
        var_type_array_orig = np.array(self.var_type_input)
        self.ordered_mask_orig = np.isin(var_type_array_orig, ["int", "num", "float"])
        self.factor_mask_orig = var_type_array_orig == "factor"

    def _set_kriging_model_feature_types(self, k_current: int) -> None:
        """
        Set variable types and masks for the Kriging model's current feature space.
        These are used when the Kriging model operates on its (possibly transformed) X_.
        """
        self.k = k_current
        # If Nyström is used, features are always numerical
        if self.approximation == "nystroem":
            self.var_type_kriging = ["num"] * k_current
            self.ordered_mask_kriging = np.array([True] * k_current)
            self.factor_mask_kriging = np.array([False] * k_current)
        else:
            # For standard Kriging, use original masks and types [7]
            self.var_type_kriging = self.var_type_input
            self.ordered_mask_kriging = self.ordered_mask_orig
            self.factor_mask_kriging = self.factor_mask_orig

    def get_model_params(self) -> Dict[str, float]:
        """ Get the model parameters (in addition to sklearn’s get_params method). """[8]
        return {"log_theta_lambda": self.logtheta_lambda_, "U": self.U_, "X": self.X_, "y": self.y_, "negLnLike": self.negLnLike}

    def _update_log(self) -> None:
        """ 
        If spot_writer is not None, this method writes the current values of negLnLike, theta, p (if optim_p is True), 
        and Lambda (if method is not "interpolation") to the spot_writer object. 
        """[
            8, 9
        ]
        self.log["negLnLike"] = append(self.log["negLnLike"], self.negLnLike)
        self.log["theta"] = append(self.log["theta"], self.theta)
        if self.optim_p:
            self.log["p"] = append(self.log["p"], self.p_val)
        if (self.method == "regression") or (self.method == "reinterpolation"):
            self.log["Lambda"] = append(self.log["Lambda"], self.Lambda)
        self.log_length = len(self.log["negLnLike"])
        if self.spot_writer is not None:
            negLnLike = self.negLnLike.copy()
            self.spot_writer.add_scalar("spot_negLnLike", negLnLike, self.counter + self.log_length)[10, 11]
            theta = self.theta.copy()
            self.spot_writer.add_scalars("spot_theta", {f"theta_{i}": theta[i] for i in range(self.n_theta)}, self.counter + self.log_length)[11, 12]
            if (self.method == "regression") or (self.method == "reinterpolation"):
                Lambda = self.Lambda.copy()
                self.spot_writer.add_scalar("spot_Lambda", Lambda, self.counter + self.log_length)[11, 12]
            if self.optim_p:
                p = self.p_val.copy()
                self.spot_writer.add_scalars("spot_p", {f"p_{i}": p[i] for i in range(self.n_p)}, self.counter + self.log_length)[11, 12]
            self.spot_writer.flush()

    def _compute_kernel_matrix_for_original_features(self, X1: np.ndarray, X2: np.ndarray, theta10_for_kernel: np.ndarray) -> np.ndarray:
        """
        Computes the kernel (correlation) matrix between two sets of ORIGINAL features X1 and X2,
        respecting original variable types (ordered/factor).
        This helper is specifically for the Nyström transformation stage.
        """
        n1 = X1.shape
        n2 = X2.shape
        D = np.zeros((n1, n2), dtype=np.float64)

        if self.ordered_mask_orig.any():
            X1_ordered = X1[:, self.ordered_mask_orig]
            X2_ordered = X2[:, self.ordered_mask_orig]
            # Ensure theta10_for_kernel has the correct length for ordered variables
            theta10_ordered = theta10_for_kernel[self.ordered_mask_orig] if theta10_for_kernel.size > 1 else theta10_for_kernel * np.ones(X1_ordered.shape[13])
            D_ordered = cdist(X1_ordered, X2_ordered, metric="sqeuclidean", w=theta10_ordered)[1, 14]
            D += D_ordered

        if self.factor_mask_orig.any():
            X1_factor = X1[:, self.factor_mask_orig]
            X2_factor = X2[:, self.factor_mask_orig]
            # Ensure theta10_for_kernel has the correct length for factor variables
            theta10_factor = theta10_for_kernel[self.factor_mask_orig] if theta10_for_kernel.size > 1 else theta10_for_kernel * np.ones(X1_factor.shape[13])
            D_factor = cdist(X1_factor, X2_factor, metric=self.metric_factorial, w=theta10_factor)[1, 14]
            D += D_factor

        return np.exp(-D)

    def fit(self, X: np.ndarray, y: np.ndarray, bounds: Optional[List[Tuple[float, float]]] = None) -> "Kriging":
        """
        Fits the Kriging model to training data X and y. This method is compatible with scikit-learn and 
        uses differential evolution to optimize the hyperparameters (log(theta)).
        If 'approximation' is set to 'nystroem', the input data X is first transformed using Nyström approximation.
        """[
            12, 15
        ]
        X_orig = np.asarray(X)
        y_orig = np.asarray(y).flatten()

        self.n_orig, k_orig_val = X_orig.shape  # Store original dimensions
        self._set_original_variable_types(k_orig_val)  # Set masks for original feature space

        # Nyström Approximation Preprocessing [16, 17]
        if self.approximation == "nystroem":
            if self.n_components_nystroem is None:
                self.n_components_nystroem = min(self.n_orig, 100)  # Default to 100 or n_orig if smaller
            if not (1 <= self.n_components_nystroem <= self.n_orig):
                raise ValueError(f"n_components_nystroem must be between 1 and n_samples ({self.n_orig}), got {self.n_components_nystroem}")

            # Sample subset for Nyström [18, 19]
            self._nystroem_X_subset_indices = self.rng.choice(self.n_orig, size=self.n_components_nystroem, replace=False)
            self._nystroem_X_subset = X_orig[self._nystroem_X_subset_indices]

            # Use initial theta (e.g., 10^0 = 1 for all original dimensions) for Nyström kernel computation
            # The actual Kriging theta will be optimized later for the Nyström features.
            initial_theta_for_nystroem = np.power(10.0, np.zeros(self.k_orig))

            # Compute K_mm (kernel matrix of sampled points)
            K_mm = self._compute_kernel_matrix_for_original_features(self._nystroem_X_subset, self._nystroem_X_subset, initial_theta_for_nystroem)

            # Regularization for K_mm to ensure numerical stability before SVD [20]
            K_mm += self._get_eps() * np.eye(K_mm.shape)

            # Perform SVD on K_mm [20, 21]
            U_mm, s_mm, _ = svd(K_mm)

            # Filter small eigenvalues and determine actual components [20]
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
                # Compute components_sqrt_inv = U_mm @ diag(1 / sqrt(s_mm)) [21]
                self._nystroem_components_sqrt_inv = U_mm[:, valid_s] @ np.diag(1.0 / np.sqrt(s_mm[valid_s]))

                # Transform original X to Nyström features
                K_nm = self._compute_kernel_matrix_for_original_features(X_orig, self._nystroem_X_subset, initial_theta_for_nystroem)
                X_nystroem = K_nm @ self._nystroem_components_sqrt_inv

                self.X_ = X_nystroem
                self.y_ = y_orig  # y remains the same
                self._set_kriging_model_feature_types(self.X_.shape[13])  # Set Kriging masks for Nyström features
                print(f"Nyström approximation applied. Original dimensions: {self.k_orig}, Nyström features: {self.k}")

        else:  # Standard Kriging (approximation is None)
            self.X_ = X_orig
            self.y_ = y_orig
            self._set_kriging_model_feature_types(self.k_orig)  # Set Kriging masks for original features
            print(f"Standard Kriging. Dimensions: {self.k}")

        self.n = self.X_.shape  # Update n for the (possibly transformed) X_ [22]

        # Kriging fitting part (operates on self.X_ which might be Nyström features) [22, 23]
        if self.isotropic:
            self.n_theta = 1
            print(f"Isotropic model: n_theta set to {self.n_theta}")
        else:
            self.n_theta = self.k  # n_theta based on current Kriging feature dimension [24, 25]
            print(f"Anisotropic model: n_theta set to {self.n_theta}")

        self.min_X = np.min(self.X_, axis=0)
        self.max_X = np.max(self.X_, axis=0)

        if bounds is None:
            if self.method == "interpolation":
                bounds = [(self.min_theta, self.max_theta)] * self.k
            else:
                bounds = [(self.min_theta, self.max_theta)] * self.k + [(self.min_Lambda, self.max_Lambda)]

        if self.optim_p:
            n_p_to_optimize = self.n_p if self.n_p == 1 else self.k  # Number of p values to optimize (1 or k)
            bounds += [(self.min_p, self.max_p)] * n_p_to_optimize[23, 25]

        self.logtheta_lambda_, _ = self.max_likelihood(bounds)[26, 27]

        # Extract optimized parameters (theta, Lambda, p_val) based on the current feature space (self.k)
        param_idx = 0
        self.theta = self.logtheta_lambda_[param_idx : param_idx + self.n_theta][26, 27]
        param_idx += self.n_theta

        if (self.method == "regression") or (self.method == "reinterpolation"):
            self.Lambda = self.logtheta_lambda_[param_idx : param_idx + 1][26, 27]
            param_idx += 1
        else:  # interpolation
            self.Lambda = None

        if self.optim_p:
            self.p_val = self.logtheta_lambda_[param_idx : param_idx + n_p_to_optimize][26, 27]
        else:
            # If not optimizing p, use the default p_val set in init or passed.
            # Make sure p_val is an array of correct size for build_Psi/psi_vec if used for weights
            if self.n_p == 1:
                self.p_val = np.array([self.p_val])
            else:  # If n_p == k for anisotropic p, but not optimizing
                self.p_val = np.array([self.p_val] * self.k)

        self.negLnLike, self.Psi_, self.U_ = self.likelihood(self.logtheta_lambda_)[26, 28]
        self._update_log()[26, 28]
        return self

    def predict(self, X: np.ndarray, return_std=False, return_val: str = "y") -> np.ndarray:
        """
        Predicts the Kriging response at a set of points X. This method is compatible with scikit-learn.
        If 'approximation' is set to 'nystroem', the input data X is first transformed using 
        the Nyström components learned during fitting.
        """[
            28, 29
        ]
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

        # Original predict logic using X_pred_for_kriging [29, 30]
        if return_std:
            predictions, std_devs = zip(*[self._pred(x_i)[:2] for x_i in X_pred_for_kriging])
            return np.array(predictions), np.array(std_devs)
        if return_val == "s":
            self.return_std = True
            predictions, std_devs = zip(*[self._pred(x_i)[:2] for x_i in X_pred_for_kriging])
            return np.array(std_devs)
        elif return_val == "all":
            self.return_std = True
            self.return_ei = True
            predictions, std_devs, eis = zip(*[self._pred(x_i) for x_i in X_pred_for_kriging])
            return np.array(predictions), np.array(std_devs), np.array(eis)
        elif return_val == "ei":
            self.return_ei = True
            predictions, eis = zip(*[(self._pred(x_i), self._pred(x_i)[31]) for x_i in X_pred_for_kriging])
            return np.array(eis)
        else:  # "y"
            predictions = [self._pred(x_i) for x_i in X_pred_for_kriging]
            return np.array(predictions)

    def build_Psi(self) -> np.ndarray:
        """ 
        Constructs a new (n x n) correlation matrix Psi to reflect new data or a change in hyperparameters. 
        Operates on the Kriging model's current feature space (self.X_).
        """[
            32, 33
        ]
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
                D_ordered = squareform(pdist(X_ordered_for_kriging, metric="sqeuclidean", w=theta10_ordered_for_kriging))[34, 35]
                Psi += D_ordered

            if self.factor_mask_kriging.any():
                X_factor_for_kriging = self.X_[:, self.factor_mask_kriging]
                theta10_factor_for_kriging = theta10[self.factor_mask_kriging]
                D_factor = squareform(pdist(X_factor_for_kriging, metric=self.metric_factorial, w=theta10_factor_for_kriging))[34, 35]
                Psi += D_factor

            Psi = np.exp(-Psi)[36, 37]
            self.inf_Psi = np.isinf(Psi).any()[36, 37]
            self.cnd_Psi = cond(Psi)[36, 37]
            return np.triu(Psi, k=1)[36, 37]
        except LinAlgError as err:
            print(f"Building Psi failed. Error: {err}, Type: {type(err)}")
            raise

    def _kernel(self, X: np.ndarray, theta: np.ndarray, p: float) -> np.ndarray:
        """
        Original _kernel method. Not directly used in the modified build_Psi path,
        but kept for completeness as per original code structure.
        """[
            36
        ]
        n_samples, n_features = X.shape
        Psi = np.zeros((n_samples, n_samples), dtype=float)
        diff = np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** p
        dist_matrix = np.sum(theta * diff, axis=2)
        Psi = np.exp(-dist_matrix)
        return np.triu(Psi, k=1)

    def likelihood(self, x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """ 
        Computes the negative of the concentrated log-likelihood for a given set of log(theta) parameters. 
        Returns the negative log-likelihood, the correlation matrix Psi, and its Cholesky factor U.
        """[
            37, 38
        ]
        X = self.X_
        y = self.y_.flatten()

        param_idx = 0
        self.theta = x[param_idx : param_idx + self.n_theta][39, 40]
        param_idx += self.n_theta

        if (self.method == "regression") or (self.method == "reinterpolation"):
            lambda_ = x[param_idx : param_idx + 1][39, 40]
            lambda_ = 10.0 ** lambda_[39, 40]
            param_idx += 1
        elif self.method == "interpolation":
            lambda_ = self.eps[40, 41]
        else:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation'")

        if self.optim_p:
            n_p_current = self.n_p if self.n_p == 1 else self.k
            self.p_val = x[param_idx : param_idx + n_p_current][40, 41]
        else:
            if self.n_p == 1:
                self.p_val = np.array([self.p_val])
            else:
                self.p_val = np.array([self.p_val] * self.k)

        n = X.shape
        one = np.ones(n)

        Psi_upper_triangle = self.build_Psi()  # build_Psi now uses current Kriging features
        Psi = Psi_upper_triangle + Psi_upper_triangle.T + np.eye(n) + np.eye(n) * lambda_[41, 42]

        try:
            U = np.linalg.cholesky(Psi)[41, 42]
        except LinAlgError:
            return self.penalty, Psi, None[42, 43]

        LnDetPsi = 2.0 * np.sum(np.log(np.abs(np.diag(U))))[42, 43]
        temp_y = np.linalg.solve(U, y)[42, 43]
        temp_one = np.linalg.solve(U, one)[42, 43]
        vy = np.linalg.solve(U.T, temp_y)[42, 43]
        vone = np.linalg.solve(U.T, temp_one)[42, 43]
        mu = (one @ vy) / (one @ vone)[42, 43]
        resid = y - one * mu[42, 43]
        tresid = np.linalg.solve(U, resid)[42, 43]
        tresid = np.linalg.solve(U.T, tresid)[42, 43]
        SigmaSqr = (resid @ tresid) / n[42, 43]
        negLnLike = (n / 2.0) * np.log(SigmaSqr) + 0.5 * LnDetPsi[2, 43]
        return negLnLike, Psi, U

    def build_psi_vec(self, x: np.ndarray) -> np.ndarray:
        """ 
        Build the psi vector required for predictive methods. 
        Operates on a single point 'x' in the Kriging model's current feature space.
        """[
            2, 43
        ]
        try:
            n = self.X_.shape
            psi = np.zeros(n)
            theta10 = np.power(10.0, self.theta)

            k_current = self.X_.shape[13]

            if self.n_theta == 1:
                theta10 = theta10 * np.ones(k_current)

            D = np.zeros(n)

            # Use Kriging model's own masks (self.ordered_mask_kriging, self.factor_mask_kriging)
            # These masks are set in _set_kriging_model_feature_types based on whether Nyström is active.
            if self.ordered_mask_kriging.any():
                X_ordered_for_kriging = self.X_[:, self.ordered_mask_kriging]
                x_ordered_for_kriging = x[self.ordered_mask_kriging]
                theta10_ordered_for_kriging = theta10[self.ordered_mask_kriging]
                D += cdist(x_ordered_for_kriging.reshape(1, -1), X_ordered_for_kriging, metric="sqeuclidean", w=theta10_ordered_for_kriging).ravel()[1, 14]

            if self.factor_mask_kriging.any():
                X_factor_for_kriging = self.X_[:, self.factor_mask_kriging]
                x_factor_for_kriging = x[self.factor_mask_kriging]
                theta10_factor_for_kriging = theta10[self.factor_mask_kriging]
                D += cdist(x_factor_for_kriging.reshape(1, -1), X_factor_for_kriging, metric=self.metric_factorial, w=theta10_factor_for_kriging).ravel()[1, 14]

            psi = np.exp(-D)[1, 44]
            return psi
        except np.linalg.LinAlgError as err:
            print(f"Building psi failed due to a linear algebra error: {err}. Error type: {type(err)}")
            raise

    def _pred(self, x: np.ndarray) -> Tuple[float, float, float]:
        """
        Computes a single-point Kriging prediction using the correlation matrix information.
        Internal helper method. [1, 44]
        """
        y = self.y_.flatten()

        if self.logtheta_lambda_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        param_idx = 0
        self.theta = self.logtheta_lambda_[param_idx : param_idx + self.n_theta][45, 46]
        param_idx += self.n_theta

        if (self.method == "regression") or (self.method == "reinterpolation"):
            lambda_ = self.logtheta_lambda_[param_idx : param_idx + 1][45, 46]
            lambda_ = 10.0 ** lambda_[45, 46]
            param_idx += 1
        elif self.method == "interpolation":
            lambda_ = self.eps[45, 47]

        if self.optim_p:
            n_p_current = self.n_p if self.n_p == 1 else self.k
            self.p_val = self.logtheta_lambda_[param_idx : param_idx + n_p_current][47, 48]
        else:
            if self.n_p == 1:
                self.p_val = np.array([self.p_val])
            else:
                self.p_val = np.array([self.p_val] * self.k)

        U = self.U_
        n = self.X_.shape
        one = np.ones(n)

        y_tilde = np.linalg.solve(U, y)[47, 48]
        y_tilde = np.linalg.solve(U.T, y_tilde)[48, 49]
        one_tilde = np.linalg.solve(U, one)[48, 49]
        one_tilde = np.linalg.solve(U.T, one_tilde)[48, 49]
        mu = (one @ y_tilde) / (one @ one_tilde)[48, 49]

        resid = y - one * mu[48, 49]
        resid_tilde = np.linalg.solve(U, resid)[48, 49]
        resid_tilde = np.linalg.solve(U.T, resid_tilde)[48, 49]

        psi = self.build_psi_vec(x)[48, 49]

        if (self.method == "interpolation") or (self.method == "regression"):
            SigmaSqr = (resid @ resid_tilde) / n[49, 50]
            psi_tilde = np.linalg.solve(U, psi)[49, 50]
            psi_tilde = np.linalg.solve(U.T, psi_tilde)[50, 51]
            SSqr = SigmaSqr * (1 + lambda_ - psi @ psi_tilde)[50, 51]
        else:  # method is "reinterpolation"
            Psi_adjusted = self.Psi_ - np.eye(n) * lambda_ + np.eye(n) * self.eps[50, 51]
            SigmaSqr = (resid @ np.linalg.solve(U.T, np.linalg.solve(U, Psi_adjusted @ resid_tilde))) / n[50, 51]
            Uint = np.linalg.cholesky(Psi_adjusted)[50, 51]
            psi_tilde = np.linalg.solve(Uint, psi)[50, 51]
            psi_tilde = np.linalg.solve(Uint.T, psi_tilde)[51, 52]
            SSqr = SigmaSqr * (1 - psi @ psi_tilde)[51, 52]

        s = np.abs(SSqr) ** 0.5[51, 52]
        f = mu + psi @ resid_tilde[51, 52]

        if self.return_ei:
            yBest = np.min(y)[51, 52]
            s_safe = s + self.eps  # Add epsilon for numerical stability
            SSqr_safe = SSqr + self.eps  # Add epsilon for numerical stability

            EITermOne = (yBest - f) * (0.5 + 0.5 * erf((1 / np.sqrt(2)) * ((yBest - f) / s_safe)))[51, 52]
            EITermTwo = s_safe * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((yBest - f) ** 2 / SSqr_safe))[51, 52]
            ExpImp = np.log10(EITermOne + EITermTwo + self.eps)  # Add epsilon to log argument
            return float(f), float(s), float(-ExpImp)
        else:
            return float(f), float(s)

    def max_likelihood(self, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        """
        Maximizes the Kriging likelihood function using differential evolution over the range of log(theta)
        specified by bounds. [52, 53]
        """

        def objective(logtheta_lambda):
            neg_ln_like, _, _ = self.likelihood(logtheta_lambda)
            return neg_ln_like

        result = differential_evolution(objective, bounds, seed=self.seed, maxiter=self.model_fun_evals)[54, 55]
        return result.x, result.fun

    def plot(self, i: int = 0, j: int = 1, show: Optional[bool] = True, add_points: bool = True) -> None:
        """
        This function plots 1D and 2D surrogates. Only for compatibility with the old Kriging implementation. [54, 55]
        """
        if self.k == 1:
            n_grid = 100
            x = linspace(self.min_X, self.max_X, num=n_grid).reshape(-1, 1)
            y = self.predict(x)
            plt.figure()
            plt.plot(x, y, "k")
            if show:
                plt.show()
        else:
            # The original source mentions 'from spotpython.surrogate.plot import plotkd'. [56, 57]
            # This implementation assumes 'plotkd' would be correctly imported and handle
            # the current (possibly transformed) self.X_ and self.var_type_kriging.
            try:
                plotkd(model=self, X=self.X_, y=self.y_, i=i, j=j, show=show, var_type=self.var_type_kriging, add_points=add_points)[58, 59]
            except NameError:
                print("`plotkd` function not found. Please ensure `spotpython.surrogate.plot` is correctly installed and imported for 2D plotting.")
                print(f"Attempted to plot dimensions {i} and {j} using Kriging model's current features of dimension {self.k}.")
