# -*- coding: utf-8 -*-
"""
This is the Kriging surrogate model.
It is based on the DACE matlab toolbox.
It can handle numerical and categorical variables.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, cho_solve, solve_triangular


class Kriging:
    """
    Kriging class with optional Nyström approximation for scalability.
    This class implements the Kriging surrogate model, also known as
    Gaussian Process regression. It is adapted to handle both numerical
    (ordered) and categorical (factor) variables, a key feature of spotpython.
    The Nyström approximation is added as an optional feature to handle
    large datasets efficiently.
    """

    def __init__(self, fun_control, n_theta=None, theta=None, p=2.0, corr="squared_exponential", isotropic=False, approximation="None", n_landmarks=100):
        """
        Initialize the Kriging model.

        Args:
            fun_control (dict): Control dictionary from spotpython, containing
                                problem dimensions, variable types ('var_type'), etc.
            n_theta (int, optional): Number of correlation parameters (theta).
                                     Defaults to problem dimension for anisotropic model.
            theta (np.ndarray, optional): Initial correlation parameters.
                                          Defaults to 0.1 for all dimensions.
            p (float, optional): Power for the correlation function. Defaults to 2.0.
            corr (str, optional): Correlation function type.
                                  Defaults to "squared_exponential".
            isotropic (bool, optional): Whether to use an isotropic model (one theta
                                        for all dimensions). Defaults to False.
            approximation (str, optional): Type of approximation to use.
                                           "None" for standard Kriging,
                                           "nystroem" for Nyström approximation.
                                           Defaults to "None".
            n_landmarks (int, optional): Number of landmark points for Nyström.
                                         Only used if approximation="nystroem".
                                         Defaults to 100.
        """
        self.fun_control = fun_control
        self.dim = self.fun_control["lower"].shape
        self.p = p
        self.corr = corr
        self.isotropic = isotropic
        self.approximation = approximation
        self.n_landmarks = n_landmarks

        # Setup masks for variable types
        self.factor_mask = self.fun_control["var_type"] == "factor"
        self.ordered_mask = ~self.factor_mask

        # Determine number of theta parameters
        if self.isotropic:
            self.n_theta = 1
        elif n_theta is None:
            self.n_theta = self.dim
        else:
            self.n_theta = n_theta

        # Initialize theta
        if theta is None:
            self.theta = np.full(self.n_theta, 0.1)
        else:
            self.theta = theta

        # Model state attributes
        self.X_ = None
        self.y_ = None
        self.L_ = None  # Cholesky factor for standard Kriging
        self.alpha_ = None  # Solved term for standard Kriging

        # Nyström-specific attributes
        self.landmarks_ = None
        self.W_cho_ = None  # Cholesky factor of W matrix
        self.nystrom_alpha_ = None  # Solved term for Nyström prediction

    def fit(self, X, y):
        """
        Fit the Kriging model to the training data.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        """
        self.X_ = X
        self.y_ = y
        n_samples = X.shape

        if self.approximation.lower() == "nystroem":
            if n_samples <= self.n_landmarks:
                # Fallback to standard Kriging if not enough samples
                self._fit_standard(X, y)
            else:
                self._fit_nystrom(X, y)
        else:
            self._fit_standard(X, y)

    def _fit_standard(self, X, y):
        """Standard Kriging fitting procedure."""
        # Build the full covariance matrix Psi
        Psi = self.build_Psi(X, X)
        Psi[np.diag_indices_from(Psi)] += 1e-8  # Add jitter for stability

        try:
            # Compute Cholesky decomposition
            self.L_ = cholesky(Psi, lower=True)
            # Solve for alpha = L'\(L\y)
            self.alpha_ = cho_solve((self.L_, True), y)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky fails
            pi_Psi = np.linalg.pinv(Psi)
            self.alpha_ = np.dot(pi_Psi, y)
            self.L_ = None  # Indicate that Cholesky failed

    def _fit_nystrom(self, X, y):
        """Nyström approximation fitting procedure."""
        n_samples = X.shape

        # 1. Select landmark points using uniform random sampling without replacement
        landmark_indices = np.random.choice(n_samples, self.n_landmarks, replace=False)
        self.landmarks_ = X[landmark_indices, :]

        # 2. Construct core matrices using build_Psi
        # W = K_mm (landmark-landmark covariance)
        W = self.build_Psi(self.landmarks_, self.landmarks_)
        W += 1e-8  # Add jitter

        # C = K_nm (data-landmark cross-covariance)
        C = self.build_Psi(X, self.landmarks_)

        # 3. Compute Cholesky decomposition of W
        try:
            self.W_cho_ = cholesky(W, lower=True)
        except np.linalg.LinAlgError:
            self.W_cho_ = None
            # Fallback to standard Kriging as a safe option
            self._fit_standard(X, y)
            return

        # 4. Pre-compute terms for prediction
        # Solve for nystrom_alpha = W_inv * C.T * y
        Ct_y = C.T @ y
        self.nystrom_alpha_ = cho_solve((self.W_cho_, True), Ct_y)

    def predict(self, X_star):
        """
        Make predictions with the fitted Kriging model.

        Args:
            X_star (np.ndarray): Test data of shape (n_test_samples, n_features).

        Returns:
            tuple: A tuple containing:
                - y_pred (np.ndarray): Predicted mean values.
                - y_mse (np.ndarray): Mean squared error (predictive variance).
        """
        if self.approximation.lower() == "nystroem" and self.landmarks_ is not None:
            return self._predict_nystrom(X_star)
        else:
            return self._predict_standard(X_star)

    def _predict_standard(self, X_star):
        """Standard Kriging prediction procedure."""
        # Build cross-covariance vector/matrix psi
        psi = self.build_Psi(X_star, self.X_)

        # Predictive mean
        y_pred = psi @ self.alpha_

        # Predictive variance
        if self.L_ is not None:
            v = solve_triangular(self.L_, psi.T, lower=True)
            y_mse = 1.0 - np.sum(v**2, axis=0)
            y_mse[y_mse < 0] = 0
        else:
            pi_Psi = np.linalg.pinv(self.build_Psi(self.X_, self.X_) + 1e-8 * np.eye(self.X_.shape))
            y_mse = 1.0 - np.sum((psi @ pi_Psi) * psi, axis=1)
            y_mse[y_mse < 0] = 0

        return y_pred, y_mse.reshape(-1, 1)

    def _predict_nystrom(self, X_star):
        """Nyström approximation prediction procedure."""
        # 1. Compute cross-covariance between test points and landmarks
        psi_star_m = self.build_Psi(X_star, self.landmarks_)

        # 2. Predictive mean
        y_pred = psi_star_m @ self.nystrom_alpha_

        # 3. Predictive variance
        if self.W_cho_ is not None:
            v = cho_solve((self.W_cho_, True), psi_star_m.T)
            quad_term = np.sum(psi_star_m * v.T, axis=1)
            y_mse = 1.0 - quad_term
            y_mse[y_mse < 0] = 0
        else:
            y_mse = np.ones(X_star.shape)  # Return max uncertainty

        return y_pred, y_mse.reshape(-1, 1)

    def build_Psi(self, X1, X2):
        """Builds the covariance matrix Psi between two sets of points."""
        n1 = X1.shape
        Psi = np.zeros((n1, X2.shape))
        for i in range(n1):
            Psi[i, :] = self.build_psi_vec(X1[i, :], X2)
        return Psi

    def build_psi_vec(self, x, X_):
        """
        Builds a covariance vector between a point x and a set of points X_.
        This method correctly handles mixed (ordered/factor) variable types.
        """
        # Handle theta for isotropic vs. anisotropic cases
        if self.isotropic:
            theta10 = np.full(self.dim, 10**self.theta)
        else:
            theta10 = 10**self.theta

        D = np.zeros(X_.shape)

        # Compute ordered distance contributions
        if self.ordered_mask.any():
            X_ordered = X_[:, self.ordered_mask]
            x_ordered = x[self.ordered_mask]
            D += cdist(x_ordered.reshape(1, -1), X_ordered, metric="sqeuclidean", w=theta10[self.ordered_mask]).ravel()

        # Compute factor distance contributions
        if self.factor_mask.any():
            X_factor = X_[:, self.factor_mask]
            x_factor = x[self.factor_mask]
            # Hamming distance for factors
            D += cdist(x_factor.reshape(1, -1), X_factor, metric="hamming", w=theta10[self.factor_mask]).ravel() * self.factor_mask.sum()

        # Apply correlation function
        if self.corr == "squared_exponential":
            psi = np.exp(-D)
        else:
            # Fallback for other potential correlation functions
            psi = np.exp(-(D**self.p))

        return psi
