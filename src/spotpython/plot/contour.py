import matplotlib.pyplot as plt
import numpy as np
import os


def simple_contour(
    fun,
    min_x=-1,
    max_x=1,
    min_y=-1,
    max_y=1,
    min_z=None,
    max_z=None,
    n_samples=100,
    n_levels=30,
):
    """
    Simple contour plot

    Args:
        fun (_type_): _description_
        min_x (int, optional): _description_. Defaults to -1.
        max_x (int, optional): _description_. Defaults to 1.
        min_y (int, optional): _description_. Defaults to -1.
        max_y (int, optional): _description_. Defaults to 1.
        min_z (int, optional): _description_. Defaults to 0.
        max_z (int, optional): _description_. Defaults to 1.
        n_samples (int, optional): _description_. Defaults to 100.
        n_levels (int, optional): _description_. Defaults to 5.

    Examples:
        >>> import matplotlib.pyplot as plt
            import numpy as np
            from spotpython.fun.objectivefunctions import analytical
            fun = analytical().fun_branin
            simple_contour(fun=fun, n_levels=30, min_x=-5, max_x=10, min_y=0, max_y=15)

    """
    XX, YY = np.meshgrid(np.linspace(min_x, max_x, n_samples), np.linspace(min_y, max_y, n_samples))
    zz = np.array([fun(np.array([xi, yi]).reshape(-1, 2)) for xi, yi in zip(np.ravel(XX), np.ravel(YY))]).reshape(n_samples, n_samples)
    fig, ax = plt.subplots(figsize=(5, 2.7), layout="constrained")
    if min_z is None:
        min_z = np.min(zz)
    if max_z is None:
        max_z = np.max(zz)
    plt.contourf(
        XX,
        YY,
        zz,
        levels=np.linspace(min_z, max_z, n_levels),
        zorder=1,
        cmap="jet",
        vmin=min_z,
        vmax=max_z,
    )
    plt.colorbar()


def plotModel(
    model,
    lower,
    upper,
    i=0,
    j=1,
    min_z=None,
    max_z=None,
    var_type=None,
    var_name=None,
    show=True,
    filename=None,
    n_grid=50,
    contour_levels=10,
    dpi=200,
    title="",
    figsize=(12, 6),
    use_min=False,
    use_max=False,
    aspect_equal=True,
    legend_fontsize=12,
    cmap="viridis",
    X_points=None,
    plot_points=True,
    points_color="white",
    points_size=30,
):
    """Generate 2D contour and 3D surface plots for a model's predictions.

    This function creates contour and 3D surface plots of a model's predictions
    over two selected dimensions (i, j). Remaining dimensions (if any) are assigned
    fixed values based on user settings (min, max, or averages).

    Args:
        model (object): A model with a predict method.
        lower (array_like): Lower bounds for each dimension.
        upper (array_like): Upper bounds for each dimension.
        i (int): Index of the dimension for the x-axis. Defaults to 0.
        j (int): Index of the dimension for the y-axis. Defaults to 1.
        min_z (float, optional): Minimum value for the color scale. Defaults to None.
        max_z (float, optional): Maximum value for the color scale. Defaults to None.
        var_type (list, optional): Variable types for each dimension. Defaults to None.
        var_name (list, optional): Variable names for labeling axes. Defaults to None.
        show (bool): Whether to display the plot. Defaults to True.
        filename (str, optional): File path to save the figure. Defaults to None.
        n_grid (int): Resolution for each dimension. Defaults to 50.
        contour_levels (int): Number of contour levels. Defaults to 10.
        dpi (int): DPI for saving the figure. Defaults to 200.
        title (str): Plot title. Defaults to "".
        figsize (tuple): Size of the figure (width, height). Defaults to (12, 6).
        use_min (bool):
            If True, hidden dimensions are set to their lower bounds.
            If both use_min and use_max are False, the mean value is used. Defaults to False.
        use_max (bool):
            If True, hidden dimensions are set to their upper bounds. Defaults to False.
            If both use_min and use_max are False, the mean value is used. Defaults to False.
        aspect_equal (bool): Whether axes have equal scaling. Defaults to True.
        legend_fontsize (int): Font size for labels and legends. Defaults to 12.
        cmap (str): Colormap for the plots. Defaults to "viridis".
        X_points: Original data points to plot.
        plot_points: Whether to plot X_points.
        points_color: Color for data points.
        points_size: Marker size for data points.

    Returns:
        tuple: (fig, axes) containing the created figure and axes objects.

    Raises:
        ValueError: If i or j are out of bounds or if lower/upper shapes mismatch.
    """
    # Validate dimensions
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    n_dims = len(lower)
    if len(upper) != n_dims:
        raise ValueError("Mismatch in dimension count between lower and upper.")
    if i >= n_dims or j >= n_dims or i < 0 or j < 0:
        raise ValueError(f"Invalid dimension indices i={i}, j={j} for {n_dims}-dimensional data.")
    if i == j:
        raise ValueError("Dimensions i and j must be different.")

    # Assign variable names if not specified
    if var_name is None:
        var_name = [f"x{k}" for k in range(n_dims)]
    elif len(var_name) != n_dims:
        raise ValueError("var_name length must match the number of dimensions.")

    # Generate x-y grid
    x_vals = np.linspace(lower[i], upper[i], n_grid)
    y_vals = np.linspace(lower[j], upper[j], n_grid)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Prepare hidden dimension strategies
    # min -> lower, max -> upper, otherwise average
    def hidden_value(dim):
        if use_min:
            return lower[dim]
        if use_max:
            return upper[dim]
        return 0.5 * (lower[dim] + upper[dim])  # average if neither min nor max

    # Construct full input points for prediction
    all_points = []
    for row in range(n_grid):
        for col in range(n_grid):
            point = np.zeros(n_dims)
            point[i] = X[row, col]
            point[j] = Y[row, col]
            for dim in range(n_dims):
                if dim != i and dim != j:
                    point[dim] = hidden_value(dim)
            all_points.append(point)
    all_points = np.array(all_points)

    # Predict
    Z_pred = model.predict(all_points)
    # Handle if the model returns dicts or tuples
    if isinstance(Z_pred, dict):
        Z_pred = Z_pred.get("mean", list(Z_pred.values())[0])
    elif isinstance(Z_pred, tuple):
        Z_pred = Z_pred[0]
    Z_pred = np.array(Z_pred).reshape(n_grid, n_grid)

    # Determine min/max Z if not given
    if min_z is None:
        min_z = np.min(Z_pred)
    if max_z is None:
        max_z = np.max(Z_pred)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # 2D contour plot
    contour = ax1.contourf(X, Y, Z_pred, levels=contour_levels, cmap=cmap, vmin=min_z, vmax=max_z)
    cbar1 = plt.colorbar(contour, ax=ax1)
    cbar1.ax.tick_params(labelsize=legend_fontsize - 2)
    ax1.set_xlabel(var_name[i], fontsize=legend_fontsize)
    ax1.set_ylabel(var_name[j], fontsize=legend_fontsize)
    ax1.tick_params(axis="both", labelsize=legend_fontsize - 2)
    if aspect_equal:
        ax1.set_aspect("equal")

    # Optionally plot original points on the 2D contour
    if plot_points and X_points is not None:
        ax1.scatter(
            X_points[:, i],
            X_points[:, j],
            c=points_color,
            edgecolor="black",
            s=points_size,
            alpha=0.9,
            zorder=5,
        )

    # 3D surface plot
    surf = ax2.plot_surface(
        X,
        Y,
        Z_pred,
        cmap=cmap,
        vmin=min_z,
        vmax=max_z,
        linewidth=0,
        antialiased=True,
        alpha=0.8,
    )
    cbar2 = fig.colorbar(surf, ax=ax2, shrink=0.7, pad=0.1)
    cbar2.ax.tick_params(labelsize=legend_fontsize - 2)
    ax2.set_xlabel(var_name[i], fontsize=legend_fontsize)
    ax2.set_ylabel(var_name[j], fontsize=legend_fontsize)
    ax2.set_zlabel("f(x)", fontsize=legend_fontsize)
    ax2.tick_params(axis="both", labelsize=legend_fontsize - 2)

    # Optionally plot original points in 3D
    if plot_points and X_points is not None:
        # Attempt model prediction for Z-values; fallback if it fails
        try:
            z_pred = model.predict(X_points)
            if isinstance(z_pred, dict):
                z_pred = z_pred.get("mean", list(z_pred.values())[0])
            elif isinstance(z_pred, tuple):
                z_pred = z_pred[0]
        except Exception:
            z_pred = np.full(X_points.shape[0], min_z)

        ax2.scatter(
            X_points[:, i],
            X_points[:, j],
            z_pred,
            c=points_color,
            edgecolor="black",
            s=points_size,
            alpha=0.9,
        )

    # Optionally set equal aspect ratio in 3D
    if aspect_equal:
        x_range = upper[i] - lower[i]
        y_range = upper[j] - lower[j]
        z_range = max_z - min_z if max_z > min_z else 1
        scale_z = (x_range + y_range) / (2 * z_range) if z_range else 1
        ax2.set_box_aspect([1, (y_range / x_range) if x_range else 1, scale_z])

    # Add title
    if title:
        fig.suptitle(title, fontsize=legend_fontsize + 2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save if requested
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=dpi)

    # Show the figure
    if show:
        plt.show()

    return fig, (ax1, ax2)


def plotCombinations(
    model,
    X=None,
    lower=None,
    upper=None,
    x_vars=None,
    y_vars=None,
    min_z=None,
    max_z=None,
    var_type=None,
    var_name=None,
    show=True,
    save_dir=None,
    n_grid=50,
    contour_levels=10,
    dpi=200,
    title_prefix="",
    figsize=(12, 6),
    use_min=False,
    use_max=False,
    margin=0.1,
    aspect_equal=False,
    legend_fontsize=12,
    cmap="viridis",
    X_points=None,
    plot_points=True,
    points_color="white",
    points_size=30,
):
    """Plot model surfaces for multiple combinations of input variables.

    This function generates contour and 3D surface plots for all specified
    combinations of input variables, avoiding redundancies and meaningless combinations.

    Args:
        model: A fitted model with a predict method.
        X: Array of training points (optional). If provided, used to derive bounds and dimension count.
        lower: Array-like lower bounds for each dimension. If None, derived from X.
        upper: Array-like upper bounds for each dimension. If None, derived from X.
        x_vars: List of indices for x-axis variables. Defaults to all if None or empty.
        y_vars: List of indices for y-axis variables. Defaults to all if None or empty.
        min_z: Min value for color scale. If None, auto-calculated.
        max_z: Max value for color scale. If None, auto-calculated.
        var_type: List of variable types. If None, assumed numeric.
        var_name: List of variable names. If None, named x0, x1, ...
        show: Whether to display the plots. Defaults to True.
        save_dir: Directory for saving plots. If None, not saved.
        n_grid: Number of grid points along each axis. Defaults to 50.
        contour_levels: Number of contour levels. Defaults to 10.
        dpi: DPI for saving figures. Defaults to 200.
        title_prefix: Prefix string for plot titles.
        figsize: Figure size (width, height). Defaults to (12, 6).
        use_min: Use lower bounds for non-plotted dimensions. Defaults to False.
        use_max: Use upper bounds for non-plotted dimensions. Defaults to False.
        margin: Fraction of range added as margin to bounds when derived from X. Defaults to 0.1.
        aspect_equal: Whether to set equal aspect ratio. Defaults to False.
        legend_fontsize: Font size for labels and legends. Defaults to 12.
        cmap (str): Colormap for the plots. Defaults to "viridis".
        X_points: Original data points to plot.
        plot_points (bool): Whether to plot X_points.
        points_color (str): Color for data points. Defaults to "white".
        points_size (int): Marker size for data points. Defaults to 30.

    Returns:
        None
    """
    # Derive bounds from X if needed
    if X is not None:
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        n_vars_X = X.shape[1]
        if lower is None:
            min_vals = np.min(X, axis=0)
            range_vals = np.ptp(X, axis=0)
            lower = min_vals - margin * range_vals
        if upper is None:
            max_vals = np.max(X, axis=0)
            range_vals = np.ptp(X, axis=0)
            upper = max_vals + margin * range_vals

    # Determine the number of variables
    if lower is not None:
        n_vars = len(lower)
    elif upper is not None:
        n_vars = len(upper)
    elif X is not None:
        n_vars = n_vars_X
    else:
        raise ValueError("Cannot determine the number of variables without X, lower, or upper.")

    # Default to all variables if x_vars or y_vars are missing
    if not x_vars:
        x_vars = list(range(n_vars))
    if not y_vars:
        y_vars = list(range(n_vars))

    # Keep track of plotted pairs
    plotted_pairs = set()

    # Generate combinations
    for i in x_vars:
        for j in y_vars:
            if i == j:
                continue
            pair = tuple(sorted([i, j]))
            if pair in plotted_pairs:
                continue
            plotted_pairs.add(pair)

            var_i_name = var_name[i] if var_name and i < len(var_name) else f"x{i}"
            var_j_name = var_name[j] if var_name and j < len(var_name) else f"x{j}"
            plot_title = f"{title_prefix}{var_i_name} vs {var_j_name}"

            filename = None
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.join(save_dir, f"plot_{var_i_name}_vs_{var_j_name}.png")

            # Call plotModel with the new arguments
            plotModel(
                model=model,
                lower=lower,
                upper=upper,
                i=i,
                j=j,
                min_z=min_z,
                max_z=max_z,
                var_type=var_type,
                var_name=var_name,
                show=show,
                filename=filename,
                n_grid=n_grid,
                contour_levels=contour_levels,
                dpi=dpi,
                title=plot_title,
                figsize=figsize,
                use_min=use_min,
                use_max=use_max,
                aspect_equal=aspect_equal,
                legend_fontsize=legend_fontsize,
                cmap=cmap,  # Pass colormap
                X_points=X_points,  # Pass original data points
                plot_points=plot_points,
                points_color=points_color,
                points_size=points_size,
            )

    return None
