import numpy as np
from numpy.linalg import LinAlgError
from typing import Dict, Tuple, List, Optional
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.special import erf
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid, array


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

    def __init__(self, eps: float = None, penalty: float = 1e4, method="regression"):
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
        self.logtheta_lambda_ = None
        self.U_ = None
        self.X_ = None
        self.y_ = None
        self.NegLnLike_ = None
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
        return {"log_theta_lambda": self.logtheta_lambda_, "U": self.U_, "X": self.X_, "y": self.y_, "NegLnLike": self.NegLnLike_}

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
            if self.method == "interpolation":
                bounds = [(-3.0, 2.0)] * k
            else:
                # regression and reinterpolation use lambda_ as well
                bounds = [(-3.0, 2.0)] * k + [(-6.0, 0.0)]

        self.logtheta_lambda_, _ = self.max_likelihood(bounds)

        # Once logtheta_lambda is found, compute the final correlation matrix
        self.NegLnLike_, self.Psi_, self.U_ = self.likelihood(self.logtheta_lambda_)
        return self

    def predict(self, X: np.ndarray, return_std=False, return_ei=False) -> np.ndarray:
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
            return_ei (bool, optional):
                If True, returns the expected improvement at each point.
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
            >>> y_pred, sd, ei = model.predict(X_test)
            >>> print("Predictions:", y_pred)
            >>> print("Standard deviations:", sd)
            >>> print("Expected improvement:", ei)
        """
        self.return_std = return_std
        self.return_ei = return_ei
        X = np.atleast_2d(X)
        if return_std and return_ei:
            # Return predictions, standard deviations, and expected improvements
            predictions, std_devs, eis = zip(*[self._pred(x_i) for x_i in X])
            return np.array(predictions), np.array(std_devs), np.array(eis)
        elif return_std:
            # Return predictions and standard deviations
            predictions, std_devs = zip(*[self._pred(x_i)[:2] for x_i in X])
            return np.array(predictions), np.array(std_devs)
        elif return_ei:
            # Return predictions and expected improvements
            predictions, eis = zip(*[(self._pred(x_i)[0], self._pred(x_i)[2]) for x_i in X])
            return np.array(predictions), np.array(eis)
        else:
            # Return only predictions
            predictions = [self._pred(x_i)[0] for x_i in X]
            return np.array(predictions)

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
                (NegLnLike, Psi, U) where:
                - NegLnLike (float): The negative concentrated log-likelihood.
                - Psi (np.ndarray): The correlation matrix.
                - U (np.ndarray): The Cholesky factor (or None if ill-conditioned).
        """
        # Extract data
        X = self.X_
        y = self.y_.flatten()

        if (self.method == "regression") or (self.method == "reinterpolation"):
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
