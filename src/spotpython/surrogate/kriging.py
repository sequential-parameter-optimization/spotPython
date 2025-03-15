import numpy as np
from numpy.linalg import LinAlgError
from typing import Dict, Tuple, List, Optional
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, RegressorMixin


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

    def __init__(self, eps: float = np.finfo(float).eps, penalty: float = 1e4, noise=True):
        """
        Initializes the Kriging model.

        Args:
            eps (float, optional):
                Small number added to the diagonal of the correlation matrix to reduce
                ill-conditioning. Defaults to machine epsilon.
            penalty (float, optional):
                Large negative log-likelihood assigned if the correlation matrix is
                not positive-definite. Defaults to 1e4.
        """
        self.eps = eps
        self.penalty = penalty
        self.logtheta_lambda_ = None
        self.U_ = None
        self.X_ = None
        self.y_ = None
        self.noise = noise

    def get_params(self, deep: bool = True) -> Dict[str, float]:
        """
        Get parameters for this estimator.

        This method is required for scikit-learn compatibility.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                contained subobjects that are estimators. Defaults to True.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return {"eps": self.eps, "penalty": self.penalty}

    def set_params(self, **params: Dict[str, float]) -> "Kriging":
        """
        Set the parameters of this estimator.

        This method is required for scikit-learn compatibility.

        Args:
            **params (dict): Estimator parameters as keyword arguments.

        Returns:
            self (Kriging): Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

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
                Bounds for each dimension of log(theta). If None, defaults to [(-3, 2)] * n_features.

        Returns:
            Kriging:
                The fitted Kriging model instance (self).

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
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        self.X_ = X
        self.y_ = y

        k = X.shape[1]
        if bounds is None:
            if self.noise:
                bounds = [(-3.0, 2.0)] * k + [(-6.0, 0.0)]
            else:
                bounds = [(-3.0, 2.0)] * k

        ModelInfo = {"X": self.X_, "y": self.y_}
        self.logtheta_lambda_, _ = self.max_likelihood(bounds, ModelInfo)

        # Once logtheta_lambda is found, compute the final correlation matrix
        NegLnLike, Psi, U = self.likelihood(self.logtheta_lambda_, ModelInfo)
        self.U_ = U

        return self

    def predict(self, X: np.ndarray, return_std=False) -> np.ndarray:
        """
        Predicts the Kriging response at a set of points X. This method is compatible
        with scikit-learn and returns predictions for the input points.

        Args:
            X (np.ndarray):
                Array of shape (n_samples, n_features) containing the points at which
                to predict the Kriging response.
            return_std (bool, optional):
                If True, returns the standard deviation of the predictions as well.
                Defaults to False.

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
            >>> y_pred, sd = model.predict(X_test)
            >>> print("Predictions:", y_pred)
            >>> print("Standard deviations:", sd)
        """
        # Create a ModelInfo dict with the final logtheta_lambda_ and U_:
        ModelInfo = {"X": self.X_, "y": self.y_, "Theta_Lambda": self.logtheta_lambda_, "U": self.U_}

        X = np.atleast_2d(X)
        if return_std:
            predictions, std_devs = zip(*[self._pred(x_i, ModelInfo) for x_i in X])
            return np.array(predictions), np.array(std_devs)
        else:
            predictions = [self._pred(x_i, ModelInfo)[0] for x_i in X]
            return np.array(predictions)

    def likelihood(self, x: np.ndarray, ModelInfo: Dict[str, np.ndarray]) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Computes the negative of the concentrated log-likelihood for a given set
        of log(theta) parameters using a power exponent p=1.99. Returns the
        negative log-likelihood, the correlation matrix Psi, and its Cholesky factor U.

        Args:
            x (np.ndarray):
                1D array of log(theta) parameters of length k. If self.noise is True,
                length is k+1 and the last element of x is the log(noise) parameter.
            ModelInfo (Dict[str, np.ndarray]):
                Contains "X" (design points) and "y" (targets).

        Returns:
            (float, np.ndarray, np.ndarray):
                (NegLnLike, Psi, U) where:
                - NegLnLike (float): The negative concentrated log-likelihood.
                - Psi (np.ndarray): The correlation matrix.
                - U (np.ndarray): The Cholesky factor (or None if ill-conditioned).
        """
        # Extract data
        X = ModelInfo["X"]
        y = ModelInfo["y"].flatten()

        if self.noise:
            theta = x[:-1]
            lambda_ = x[-1]
        else:
            theta = x
            lambda_ = self.eps

        theta = 10.0**theta
        lambda_ = 10.0**lambda_
        p = 1.99
        n = X.shape[0]
        one = np.ones(n)

        # Build correlation matrix
        Psi = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                dist_vec = np.abs(X[i, :] - X[j, :]) ** p
                Psi[i, j] = np.exp(-np.sum(theta * dist_vec))

        Psi = Psi + Psi.T + np.eye(n) + np.eye(n) * lambda_

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

        NegLnLike = (n / 2.0) * np.log(SigmaSqr) + 0.5 * LnDetPsi
        return NegLnLike, Psi, U

    def _pred(self, x: np.ndarray, ModelInfo: Dict[str, np.ndarray]) -> float:
        """
        Computes a single-point Kriging prediction using the correlation matrix
        information in ModelInfo. Internal helper method.

        Args:
            x (np.ndarray): 1D array of length k for the point at which to predict.
            ModelInfo (Dict[str, np.ndarray]): Must contain "X", "y", "Theta_Lambda", and "U".

        Returns:
            float: The Kriging prediction at x.
            float: The standard deviation of the prediction.
        """
        X = ModelInfo["X"]
        y = ModelInfo["y"].flatten()

        if self.noise:
            theta = ModelInfo["Theta_Lambda"][:-1]
            lambda_ = ModelInfo["Theta_Lambda"][-1]
        else:
            theta = ModelInfo["Theta_Lambda"]
            lambda_ = self.eps

        theta = 10.0**theta
        lambda_ = 10.0**lambda_

        U = ModelInfo["U"]

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

        # Compute SigmaSqr
        SigmaSqr = (resid @ resid_tilde) / n
        # Compute SSqr
        psi_tilde = np.linalg.solve(U, psi)
        psi_tilde = np.linalg.solve(U.T, psi_tilde)
        SSqr = SigmaSqr * (1 + lambda_ - psi @ psi_tilde)
        # Compute s
        s = np.abs(SSqr) ** 0.5

        # Final prediction
        # TODO: check igf the following is can be omitted:
        # resid = y - one * mu
        # resid_tilde = np.linalg.solve(U, resid)
        # resid_tilde = np.linalg.solve(U.T, resid_tilde)
        f = mu + psi @ resid_tilde

        return float(f), float(s)

    def max_likelihood(self, bounds: List[Tuple[float, float]], ModelInfo: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Maximizes the Kriging likelihood function using differential evolution
        over the range of log(theta) specified by bounds.

        Args:
            bounds (List[Tuple[float, float]]): Sequence of (low, high) bounds for log(theta).
            ModelInfo (Dict[str, np.ndarray]): The model data with "X" and "y".

        Returns:
            (np.ndarray, float): (best_x, best_fun) where best_x is the
            optimal log(theta) array and best_fun is the minimized negative log-likelihood.
        """

        def objective(logtheta_lambda):
            neg_ln_like, _, _ = self.likelihood(logtheta_lambda, ModelInfo)
            return neg_ln_like

        result = differential_evolution(objective, bounds)
        return result.x, result.fun
