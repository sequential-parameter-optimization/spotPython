import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid
from typing import Optional, List
import matplotlib
import pylab


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


def generate_mesh_grid(X: np.ndarray, i: int, j: int, n_grid: int = 100):
    """
    Generate a mesh grid for two selected dimensions of X, and fill the remaining dimensions with their mean values.

    Args:
        X (np.ndarray): Input data of shape (n_samples, k).
        i (int): Index of the first dimension to vary.
        j (int): Index of the second dimension to vary.
        n_grid (int): Number of grid points per dimension.

    Returns:
        X_i (np.ndarray): Meshgrid for the i-th dimension.
        X_j (np.ndarray): Meshgrid for the j-th dimension.
        grid_points (np.ndarray): Grid points of shape (n_grid*n_grid, k) for prediction.
    """
    k = X.shape[1]
    mean_values = X.mean(axis=0)

    # Create a grid for the two varied dimensions
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

    return X_i, X_j, grid_points


def plot_values(
    ax: "matplotlib.axes.Axes",
    X: np.ndarray,
    y: np.ndarray,
    model,
    i: int,
    j: int,
    Z,
    surface_label: str = "Prediction Surface",
    zlabel: str = "Prediction",
    var_names: Optional[List[str]] = None,
    alpha: float = 0.8,
    eps: float = 1e-3,
    cmap: str = "jet",
    error_surface: bool = False,
) -> None:
    """
    Plot a 3D surface and scatter input points, colored by prediction error.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib 3D axis.
        X (np.ndarray): Input data, shape (n_samples, k).
        y (np.ndarray): Target values, shape (n_samples,).
        model (object): Fitted model with predict().
        i (int): Index of first varied dimension.
        j (int): Index of second varied dimension.
        Z (tuple or np.ndarray): Surface values to plot, shape matching meshgrid.
        surface_label (str): Title for the surface.
        zlabel (str): Label for the z-axis.
        var_names (list of str or None): List of axis labels or None.
        alpha (float): Surface transparency.
        eps (float): Tolerance for error coloring.
        cmap (str): Colormap for the surface.
        error_surface (bool): If True, scatter z is abs(y_actual - y_predicted).

    Returns:
        None
    """
    ax.plot_surface(*Z[:2], Z[2], cmap=cmap, alpha=alpha) if isinstance(Z, tuple) else ax.plot_surface(Z[0], Z[1], Z[2], cmap=cmap, alpha=alpha)
    ax.set_title(surface_label)
    ax.set_xlabel(var_names[0] if var_names else f"Dimension {i}")
    ax.set_ylabel(var_names[1] if var_names else f"Dimension {j}")
    ax.set_zlabel(var_names[2] if var_names else zlabel)
    for idx in range(X.shape[0]):
        x_point = X[idx, i]
        y_point = X[idx, j]
        z_actual = y[idx]
        z_predicted = model.predict(X[idx].reshape(1, -1))[0]
        if error_surface:
            z_scatter = abs(z_actual - z_predicted)
        else:
            z_scatter = z_actual
        if z_actual > z_predicted + eps:
            color = "red"
        elif z_actual < z_predicted - eps:
            color = "green"
        else:
            color = "white"
        ax.scatter(x_point, y_point, z_scatter, color=color, s=50, edgecolor="black")


def plotkd(
    model,
    X: np.ndarray,
    y: np.ndarray,
    i: int = 0,
    j: int = 1,
    show: Optional[bool] = True,
    alpha=0.8,
    eps=1e-3,
    var_names: Optional[List[str]] = None,
    cmap="jet",
    n_grid: int = 100,
) -> None:
    """
    Plots the Kriging surrogate model for k-dimensional input data by varying two dimensions (i, j).

    Args:
        model (object): A fitted Kriging model.
        X (np.ndarray): Training input data of shape (n_samples, k).
        y (np.ndarray): Training target values of shape (n_samples,).
        i (int): Index of the first dimension to vary. Default is 0.
        j (int): Index of the second dimension to vary. Default is 1.
        show (bool): If True, displays the plot. Default is True.
        alpha (float): Transparency of the surface plot. Default is 0.8.
        eps (float): Tolerance for coloring points based on prediction error. Default is 1e-3.
        var_names (list of str, optional): List of variable names for axis labeling. If None, generic labels are used.
        cmap (str): Colormap for the surface and contour plots. Default is "jet".
        n_grid (int): Number of grid points per dimension for the mesh grid. Default is 100.
    """
    k = X.shape[1]
    if i >= k or j >= k:
        raise ValueError(f"Dimensions i and j must be less than the number of features (k={k}).")
    if i == j:
        raise ValueError("Dimensions i and j must be different.")

    X_i, X_j, grid_points = generate_mesh_grid(X, i, j, n_grid)

    # Predict the values and standard deviations
    y_pred, y_std = model.predict(grid_points, return_std=True)
    Z_pred = y_pred.reshape(X_i.shape)
    Z_std = y_std.reshape(X_i.shape)

    fig = plt.figure(figsize=(12, 10))

    # Plot predicted values
    ax1 = fig.add_subplot(221, projection="3d")
    plot_values(
        ax1,
        X,
        y,
        model,
        i,
        j,
        (X_i, X_j, Z_pred),
        surface_label="Prediction Surface",
        zlabel="Prediction",
        var_names=var_names,
        alpha=alpha,
        eps=eps,
        cmap=cmap,
        error_surface=False,
    )

    # Plot prediction error
    ax2 = fig.add_subplot(222, projection="3d")
    plot_values(
        ax2,
        X,
        y,
        model,
        i,
        j,
        (X_i, X_j, Z_std),
        surface_label="Prediction Error Surface",
        zlabel="Error",
        var_names=var_names,
        alpha=alpha,
        eps=eps,
        cmap=cmap,
        error_surface=True,
    )

    # Contour plot of predicted values
    ax3 = fig.add_subplot(223)
    contour = ax3.contourf(X_i, X_j, Z_pred, cmap=cmap, levels=30)
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
    contour = ax4.contourf(X_i, X_j, Z_std, cmap=cmap, levels=30)
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


def plot_3d_contour(X, Y, Z, vmin, vmax, var_name=None, i=0, j=1, show=True, filename=None, contour_levels=10, dpi=200, title=None, figsize=(12, 6), tkagg=False, cmap="jet") -> None:
    """
    Plots a 3D surface and filled contour for a surrogate model's predictions over a grid.

    Args:
        X (np.ndarray): 2D array of x-coordinates for the grid.
        Y (np.ndarray): 2D array of y-coordinates for the grid.
        Z (np.ndarray): 2D array of z-coordinates (predictions) for the grid.
        vmin (float): Minimum value for the color scale.
        vmax (float): Maximum value for the color scale.
        var_name (list or None): List of variable names for axis labeling. If None, generic labels are used.
        i (int, optional): Index of the first variable to plot. Default is 0.
        j (int, optional): Index of the second variable to plot. Default is 1.
        show (bool, optional): If True, displays the plot interactively. Default is True.
        filename (str, optional): If provided, saves the plot to this file. Default is None.
        contour_levels (int, optional): Number of contour levels in the filled contour plot. Default is 10.
        dpi (int, optional): Dots per inch for saved figure. Default is 200.
        title (str, optional): Title for the plot. Default is None.
        figsize (tuple, optional): Figure size in inches (width, height). Default is (12, 6).
        tkagg (bool, optional): If True, use TkAgg backend for matplotlib. Default is False.
        cmap (str, optional): Colormap for the surface and contour plots. Default is "jet".

    Returns:
        None

    Examples:
        >>> # Example 1: Using output from Spot
        >>> # Assume S is a Spot object with a fitted surrogate
        >>> plot_data = S.prepare_plot(i=0, j=1, n_grid=100)
        >>> from spotpython.surrogate.plot import plot_3d_contour
        >>> plot_3d_contour(
        ...     plot_data,
        ...     var_name=S.var_name,
        ...     i=0,
        ...     j=1,
        ...     title="Surrogate Model Contour",
        ...     contour_levels=25,
        ...     show=True
        ... )
        >>> # Example 2: Using plot_3d_contour from scratch
        >>> import numpy as np
        >>> from spotpython.surrogate.plot import plot_3d_contour
        >>> # Create a grid
        >>> x = np.linspace(-5, 5, 100)
        >>> y = np.linspace(-5, 5, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> # Define a function over the grid
        >>> Z = np.sin(np.sqrt(X**2 + Y**2))
        >>> plot_data = {
        ...     "X_combined": X,
        ...     "Y_combined": Y,
        ...     "Z_combined": Z,
        ...     "min_z": Z.min(),
        ...     "max_z": Z.max(),
        ... }
        >>> plot_3d_contour(
        ...     plot_data,
        ...     var_name=["x", "y"],
        ...     i=0,
        ...     j=1,
        ...     title="Sine Surface",
        ...     contour_levels=20,
        ...     show=True
        ... )
    """
    if tkagg:
        matplotlib.use("TkAgg")
    fig = pylab.figure(figsize=figsize)

    ax_3d = fig.add_subplot(121, projection="3d")
    ax_3d.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.9, cmap=cmap, vmin=vmin, vmax=vmax)
    set_contour_labels(ax_3d, i=i, j=j, var_name=var_name, title=title)

    ax_contour = fig.add_subplot(122)
    contour = ax_contour.contourf(X, Y, Z, levels=contour_levels, zorder=1, cmap=cmap, vmin=vmin, vmax=vmax)
    pylab.colorbar(contour, ax=ax_contour)
    set_contour_labels(ax_contour, i=i, j=j, var_name=var_name, title=title)

    if filename:
        pylab.savefig(filename, bbox_inches="tight", dpi=dpi, pad_inches=0)
    if show:
        pylab.show()


def set_contour_labels(ax, i=0, j=1, var_name=None, title=None) -> None:
    """
    Set axis labels and title for a contour plot.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib Axes object to label.
        i (int): Index of the first variable (x-axis).
        j (int): Index of the second variable (y-axis).
        var_name (list or None): List of variable names, or None for generic labels.
        title (str or None): Title for the plot, or None.

    Returns:
        None
    """
    if var_name is None:
        ax.set_xlabel(f"x{i}")
        ax.set_ylabel(f"x{j}")
    else:
        ax.set_xlabel(f"x{i}: {var_name[i]}")
        ax.set_ylabel(f"x{j}: {var_name[j]}")
    if title is not None:
        ax.set_title(title)
