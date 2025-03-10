import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pylab
import os


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
    use_max=True,
    margin=0.1,
    aspect_equal=False,
    legend_fontsize=12,
):
    """Plot model surfaces for multiple combinations of input variables.

    This function generates contour and 3D surface plots for all specified combinations
    of input variables, avoiding redundant and meaningless combinations.

    Args:
        model: A fitted model object with a predict method.
        X: Array of input points used for training. If provided, variable count and
            bounds will be derived from this matrix when not explicitly specified.
        lower: Array-like with lower bounds for all dimensions. If None, derived
            from X. Defaults to None.
        upper: Array-like with upper bounds for all dimensions. If None, derived
            from X. Defaults to None.
        x_vars: List of indices for x-axis variables. If None or empty, all variables
            are used. Defaults to None.
        y_vars: List of indices for y-axis variables. If None or empty, all variables
            are used. Defaults to None.
        min_z: Minimum value for the colorbar. If None, determined from the data.
        max_z: Maximum value for the colorbar. If None, determined from the data.
        var_type: List of variable types ("float", "int", etc.). If None, all types are
            assumed to be numeric.
        var_name: List of variable names. If None, generic names x0, x1, ... are used.
        show: Whether to display the plots. Defaults to True.
        save_dir: Directory to save plots. If provided, plots are saved as PNG files.
            Defaults to None.
        n_grid: Number of grid points in each dimension. Defaults to 50.
        contour_levels: Number of contour levels. Defaults to 10.
        dpi: DPI for saving figures. Defaults to 200.
        title_prefix: Prefix for plot titles. Defaults to "".
        figsize: Figure size as (width, height) tuple. Defaults to (12, 6).
        use_min: Whether to use minimum values for non-plotted dimensions. Defaults to False.
        use_max: Whether to use maximum values for non-plotted dimensions. Defaults to True.
        margin: Fraction of range to add as margin when determining bounds from X.
            Defaults to 0.1 (10%).
        aspect_equal: Whether to set the aspect ratio to 1:1. Defaults to False.
        legend_fontsize: Font size for the legend and plot labels. Defaults to 12.

    Returns:
        None. Displays and/or saves the visualizations.

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.gp_sep import GPsep
        >>> from spotpython.utils.plot_utils import plotCombinations
        >>>
        >>> # Create and fit a model
        >>> X = np.random.rand(30, 5)
        >>> y = np.sum(X**2, axis=1)
        >>> model = GPsep().fit(X, y)
        >>>
        >>> # Plot with automatic bound detection
        >>> plotCombinations(
        >>>     model=model,
        >>>     X=X,
        >>>     x_vars=[0, 2],
        >>>     y_vars=[1, 3, 4]
        >>> )
        >>>
        >>> # Plot with explicit bounds
        >>> plotCombinations(
        >>>     model=model,
        >>>     lower=np.zeros(5),
        >>>     upper=np.ones(5)
        >>> )
    """
    # If X is provided, extract number of variables and bounds if needed
    if X is not None:
        if hasattr(X, "to_numpy"):  # Handle pandas DataFrame
            X = X.to_numpy()

        # Get number of features
        n_vars_X = X.shape[1]

        # Calculate bounds if not provided
        if lower is None:
            min_vals = np.min(X, axis=0)
            # Add some margin to ensure data points aren't on the boundary
            range_vals = np.ptp(X, axis=0)  # peak-to-peak range
            lower = min_vals - margin * range_vals

        if upper is None:
            max_vals = np.max(X, axis=0)
            # Add some margin to ensure data points aren't on the boundary
            range_vals = np.ptp(X, axis=0)  # peak-to-peak range
            upper = max_vals + margin * range_vals

    # Determine number of variables from lower/upper bounds
    if lower is not None:
        n_vars = len(lower)
    elif upper is not None:
        n_vars = len(upper)
    elif X is not None:
        n_vars = n_vars_X
    else:
        raise ValueError("Either X, lower, or upper must be provided to determine the number of variables")

    # If x_vars or y_vars is None or empty, use all variable indices
    if x_vars is None or len(x_vars) == 0:
        x_vars = list(range(n_vars))
    if y_vars is None or len(y_vars) == 0:
        y_vars = list(range(n_vars))

    # Create a list to track which combinations have been plotted
    plotted_pairs = set()

    # Iterate through all combinations
    for i in x_vars:
        for j in y_vars:
            # Skip if i == j (meaningless comparison)
            if i == j:
                continue

            # Create a canonical representation of the pair to avoid redundancies
            pair = tuple(sorted([i, j]))

            # Skip if this pair has already been plotted
            if pair in plotted_pairs:
                continue

            # Add the pair to the set of plotted pairs
            plotted_pairs.add(pair)

            # Create a descriptive title
            variable_i = var_name[i] if var_name is not None else f"x{i}"
            variable_j = var_name[j] if var_name is not None else f"x{j}"
            plot_title = f"{title_prefix}{variable_i} vs {variable_j}"

            # Determine filename if saving is requested
            filename = None
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.join(save_dir, f"plot_{variable_i}_vs_{variable_j}.png")

            # Call plotModel for this combination
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
            )

    return None


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
    use_max=True,
    tkagg=False,
    aspect_equal=True,
    legend_fontsize=12,
):
    """Plot 2D contour and 3D surface for any model with a predict method.

    This function creates visualizations of a model's predictions over a 2D slice of the input
    space, showing both a contour plot and a 3D surface. It works with any model that
    implements a predict method (such as GPsep or scikit-learn models).

    Args:
        model: A fitted model object with a predict method.
        lower: Array-like with lower bounds for all dimensions.
        upper: Array-like with upper bounds for all dimensions.
        i: Index of first dimension to plot (x-axis). Defaults to 0.
        j: Index of second dimension to plot (y-axis). Defaults to 1.
        min_z: Minimum value for the colorbar. If None, determined from the data.
        max_z: Maximum value for the colorbar. If None, determined from the data.
        var_type: List of variable types ("float", "int", etc.). If None, all types are
            assumed to be numeric.
        var_name: List of variable names. If None, generic names x0, x1, ... are used.
        show: Whether to display the plot. Defaults to True.
        filename: If provided, the plot is saved to this file. Defaults to None.
        n_grid: Number of grid points in each dimension. Defaults to 50.
        contour_levels: Number of contour levels. Defaults to 10.
        dpi: DPI for saving the figure. Defaults to 200.
        title: Title for the plot. Defaults to "".
        figsize: Figure size as (width, height) tuple. Defaults to (12, 6).
        use_min: Whether to use minimum values for non-plotted dimensions. Defaults to False.
        use_max: Whether to use maximum values for non-plotted dimensions. Defaults to True.
        tkagg: Whether to use TkAgg backend for matplotlib. Defaults to False.
        aspect_equal: Whether to set aspect ratio to be equal. Defaults to True.
        legend_fontsize: Font size for the legend. Defaults to 12.

    Returns:
        None. Displays and/or saves the visualization.

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.gp_sep import GPsep
        >>> from spotpython.plot.contour import plotModel
        >>>
        >>> # Create and fit a GPsep model
        >>> X = np.random.rand(20, 3)
        >>> y = np.sum(X**2, axis=1)
        >>> model = GPsep().fit(X, y)
        >>>
        >>> # Plot the first two dimensions
        >>> plotModel(
        >>>     model=model,
        >>>     lower=np.zeros(3),
        >>>     upper=np.ones(3),
        >>>     i=0,
        >>>     j=1,
        >>>     var_name=["x", "y", "z"],
        >>>     title="Response Surface"
        >>> )
    """

    # Helper functions for the visualization
    def generate_mesh_grid(lower, upper, grid_points):
        """Generate a mesh grid for the given range."""
        x = np.linspace(lower[i], upper[i], num=grid_points)
        y = np.linspace(lower[j], upper[j], num=grid_points)
        return np.meshgrid(x, y), x, y

    def process_var_values(z00, var_type, use_min=True):
        """Process each entry according to variable type."""
        result = []
        for k in range(len(var_type)) if var_type is not None else range(len(z00[0])):
            if var_type is not None and var_type[k] == "float":
                mean_value = np.mean(z00[:, k])
                result.append(mean_value)
            else:  # For int, factor, or when var_type is None
                if use_min:
                    min_value = min(z00[:, k])
                    result.append(min_value)
                else:
                    max_value = max(z00[:, k])
                    result.append(max_value)
        return result

    def change_values(x, y, z0, idx_i, idx_j):
        """Change the values at indices i and j in z0 to x and y."""
        z0_copy = z0.copy()
        z0_copy[idx_i] = x
        z0_copy[idx_j] = y
        return z0_copy

    def plot_contour_subplots(X, Y, Z, ax, min_z, max_z, contour_levels):
        """Plot the contour and colorbar on the given axes."""
        contour = ax.contourf(X, Y, Z, contour_levels, zorder=1, cmap="jet", vmin=min_z, vmax=max_z)
        cbar = pylab.colorbar(contour, ax=ax)
        cbar.ax.tick_params(labelsize=legend_fontsize - 2)  # Adjust colorbar tick size

    # Set the matplotlib backend if needed
    if tkagg:
        matplotlib.use("TkAgg")

    # Create the figure
    fig = plt.figure(figsize=figsize)

    # Generate the mesh grid
    (X, Y), x, y = generate_mesh_grid(lower, upper, n_grid)

    # Initialize lists to store predictions
    Z_list, X_list, Y_list = [], [], []

    # Create initial parameter vectors from bounds
    z00 = np.array([lower, upper])

    # If var_type is not provided, assume all numeric
    if var_type is None:
        var_type = ["float"] * len(lower)

    # Process values for hidden dimensions based on flags
    if use_min:
        z0_min = process_var_values(z00, var_type, use_min=True)
        # Get predictions on grid with min values for hidden dimensions
        Z_min = np.zeros_like(X)
        for idx_x in range(X.shape[0]):
            for idx_y in range(X.shape[1]):
                point = change_values(X[idx_x, idx_y], Y[idx_x, idx_y], z0_min, i, j)
                # Handle different model.predict() return formats
                prediction = model.predict(np.array([point]))
                if isinstance(prediction, dict):
                    # For models that return dictionaries (like GPsep)
                    Z_min[idx_x, idx_y] = prediction["mean"] if "mean" in prediction else prediction.get("y", 0)
                elif isinstance(prediction, tuple):
                    # For models that return (mean, std_dev) tuples
                    Z_min[idx_x, idx_y] = prediction[0]
                else:
                    # For models that return direct predictions
                    Z_min[idx_x, idx_y] = prediction

        Z_list.append(Z_min)
        X_list.append(X)
        Y_list.append(Y)

    if use_max:
        z0_max = process_var_values(z00, var_type, use_min=False)
        # Get predictions on grid with max values for hidden dimensions
        Z_max = np.zeros_like(X)
        for idx_x in range(X.shape[0]):
            for idx_y in range(X.shape[1]):
                point = change_values(X[idx_x, idx_y], Y[idx_x, idx_y], z0_max, i, j)
                # Handle different model.predict() return formats
                prediction = model.predict(np.array([point]))
                if isinstance(prediction, dict):
                    # For models that return dictionaries (like GPsep)
                    Z_max[idx_x, idx_y] = prediction["mean"] if "mean" in prediction else prediction.get("y", 0)
                elif isinstance(prediction, tuple):
                    # For models that return (mean, std_dev) tuples
                    Z_max[idx_x, idx_y] = prediction[0]
                else:
                    # For models that return direct predictions
                    Z_max[idx_x, idx_y] = prediction

        Z_list.append(Z_max)
        X_list.append(X)
        Y_list.append(Y)

    # Combine predictions for visualization
    if Z_list:  # Ensure that there is at least one Z to stack
        Z_combined = np.vstack(Z_list)
        X_combined = np.vstack(X_list)
        Y_combined = np.vstack(Y_list)

        # Set min/max values for colorbar if not provided
        if min_z is None:
            min_z = np.min(Z_combined)
        if max_z is None:
            max_z = np.max(Z_combined)

        # Create contour plot
        ax_contour = fig.add_subplot(121)
        plot_contour_subplots(X_combined, Y_combined, Z_combined, ax_contour, min_z, max_z, contour_levels)

        # Set equal aspect ratio if requested
        if aspect_equal:
            ax_contour.set_aspect("equal")

        # Add axis labels
        if var_name is None:
            ax_contour.set_xlabel(f"x{i}", fontsize=legend_fontsize)
            ax_contour.set_ylabel(f"x{j}", fontsize=legend_fontsize)
        else:
            ax_contour.set_xlabel(f"x{i}: {var_name[i]}", fontsize=legend_fontsize)
            ax_contour.set_ylabel(f"x{j}: {var_name[j]}", fontsize=legend_fontsize)

        # Adjust tick label size
        ax_contour.tick_params(axis="both", which="major", labelsize=legend_fontsize - 2)

        # Create 3D surface plot
        ax_3d = fig.add_subplot(122, projection="3d")
        surf = ax_3d.plot_surface(X_combined, Y_combined, Z_combined, rstride=3, cstride=3, alpha=0.9, cmap="jet", vmin=min_z, vmax=max_z)

        # Add a colorbar for the 3D plot that's properly sized
        cbar = fig.colorbar(surf, ax=ax_3d, shrink=0.7, pad=0.1)
        cbar.ax.tick_params(labelsize=legend_fontsize - 2)

        # Set equal aspect ratio for 3D plot if requested
        if aspect_equal:
            # Calculate the ranges for scaling
            x_range = upper[i] - lower[i]
            y_range = upper[j] - lower[j]
            z_range = max_z - min_z

            # Set box aspect with scaling to make x:y ratio = 1:1
            # For z-axis, scale based on the ranges to get a reasonable height
            scale_z = (x_range + y_range) / (2 * z_range) if z_range > 0 else 1
            ax_3d.set_box_aspect([1, y_range / x_range if x_range > 0 else 1, scale_z])

        # Add axis labels to 3D plot with adjusted font size
        if var_name is None:
            ax_3d.set_xlabel(f"x{i}", fontsize=legend_fontsize)
            ax_3d.set_ylabel(f"x{j}", fontsize=legend_fontsize)
            ax_3d.set_zlabel("f(x)", fontsize=legend_fontsize)
        else:
            ax_3d.set_xlabel(f"x{i}: {var_name[i]}", fontsize=legend_fontsize)
            ax_3d.set_ylabel(f"x{j}: {var_name[j]}", fontsize=legend_fontsize)
            ax_3d.set_zlabel("f(x)", fontsize=legend_fontsize)

        # Adjust tick label size for 3D plot
        ax_3d.tick_params(axis="both", which="major", labelsize=legend_fontsize - 2)

        # Add title with proper font size
        plt.suptitle(title, fontsize=legend_fontsize + 2)

        # Adjust layout to make room for labels
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure if filename is provided
        if filename:
            pylab.savefig(filename, bbox_inches="tight", dpi=dpi, pad_inches=0)

        # Show figure if requested
        if show:
            pylab.show()


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
