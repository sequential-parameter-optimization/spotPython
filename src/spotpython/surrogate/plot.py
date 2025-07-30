import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid, array
from typing import Optional, List


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
        >>> from spotpython.surrogate.kriging import Kriging
        >>> from spotpython.surrogate.plot import plotkd
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
