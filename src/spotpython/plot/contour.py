import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib import gridspec


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
) -> None:
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

    Returns:
        None

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
    y_points=None,
    plot_points=True,
    points_color="white",
    points_size=30,
    point_color_below="blue",
    point_color_above="red",
    atol=1e-6,
) -> None:
    """
    Generate 2D contour and 3D surface plots for a model's predictions.

    Even if the data is not strictly 3D, each point in X_points will have its
    "predicted surface z-value" computed by:
      1) Taking the i-th and j-th coordinates directly from that point.
      2) Setting all other dimensions (leftover dims) based on use_min, use_max,
         or their mean (if both are False).
    Then, we compare the newly-computed "actual z" for that point with its
    model-predicted z-value. If 'actual z' < 'predicted z', the point is colored
    with point_color_below; otherwise, it is colored with point_color_above.

    Args:
        model (object): A model with a predict method.
        lower (array_like): Lower bounds for each dimension.
        upper (array_like): Upper bounds for each dimension.
        i (int): Index for the x-axis dimension.
        j (int): Index for the y-axis dimension.
        min_z (float, optional): Min value for color scaling. Defaults to None.
        max_z (float, optional): Max value for color scaling. Defaults to None.
        var_type (list, optional): Variable types for each dimension. Defaults to None.
        var_name (list, optional): Variable names for labeling axes. Defaults to None.
        show (bool): Whether to display the plot. Defaults to True.
        filename (str, optional): File path to save the figure. Defaults to None.
        n_grid (int): Resolution for each axis. Defaults to 50.
        contour_levels (int): Number of contour levels. Defaults to 10.
        dpi (int): DPI for saving. Defaults to 200.
        title (str): Title for the figure. Defaults to "".
        figsize (tuple): Figure size. Defaults to (12, 6).
        use_min (bool): If True, leftover dims are set to lower bounds.
        use_max (bool): If True, leftover dims are set to upper bounds.
        aspect_equal (bool): Whether axes have equal scaling. Defaults to True.
        legend_fontsize (int): Font size for labels and legends. Defaults to 12.
        cmap (str): Colormap. Defaults to "viridis".
        X_points (ndarray): Original data points. Shape: (N, D).
        y_points (ndarray): Original target values. Shape: (N,).
        plot_points (bool): Whether to plot X_points. Defaults to True.
        points_color (str): Fallback color for data points. Defaults to "white".
        points_size (int): Marker size for data points. Defaults to 30.
        point_color_below (str): Color if actual z < predicted z. Defaults to "lightgrey".
        point_color_above (str): Color if actual z >= predicted z. Defaults to "white".
        atol (float): Absolute tolerance for comparing actual and predicted z-values. Defaults to 1e-6.

    Returns:
        (fig, (ax_contour, ax_surface)): Figure and axes for the contour and surface plots.

    Raises:
        ValueError: For mismatched dimensions or invalid i/j indices.
    """
    # --- Validate inputs ---
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    n_dims = len(lower)
    if len(upper) != n_dims:
        raise ValueError("Mismatch in dimension count between lower and upper.")
    if i < 0 or j < 0 or i >= n_dims or j >= n_dims:
        raise ValueError(f"Invalid dimension indices i={i} or j={j} for {n_dims}-dimensional data.")
    if i == j:
        raise ValueError("Dimensions i and j must be different.")

    if var_name is None:
        var_name = [f"x{k}" for k in range(n_dims)]
    elif len(var_name) != n_dims:
        raise ValueError("var_name length must match the number of dimensions.")

    # --- 2D grid for contour/surface ---
    x_vals = np.linspace(lower[i], upper[i], n_grid)
    y_vals = np.linspace(lower[j], upper[j], n_grid)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    # Helper for leftover dims
    def hidden_value(dim_index):
        if use_min:
            return lower[dim_index]
        if use_max:
            return upper[dim_index]
        return 0.5 * (lower[dim_index] + upper[dim_index])

    # Build all grid points
    grid_points = []
    for row in range(n_grid):
        for col in range(n_grid):
            p = np.zeros(n_dims)
            p[i] = X_grid[row, col]
            p[j] = Y_grid[row, col]
            for dim in range(n_dims):
                if dim not in (i, j):
                    p[dim] = hidden_value(dim)
            grid_points.append(p)
    grid_points = np.array(grid_points)

    # Predict for the grid
    Z_pred = model.predict(grid_points)
    if isinstance(Z_pred, dict):
        Z_pred = Z_pred.get("mean", list(Z_pred.values())[0])
    elif isinstance(Z_pred, tuple):
        Z_pred = Z_pred[0]
    Z_pred = Z_pred.reshape(n_grid, n_grid)

    # Determine min/max color scale
    if min_z is None:
        min_z = np.min(Z_pred)
    if max_z is None:
        max_z = np.max(Z_pred)

    # --- Set up figure ---
    fig = plt.figure(figsize=figsize)
    ax_contour = fig.add_subplot(1, 2, 1)
    ax_surface = fig.add_subplot(1, 2, 2, projection="3d")

    # --- 2D contour ---
    cont = ax_contour.contourf(
        X_grid,
        Y_grid,
        Z_pred,
        levels=contour_levels,
        cmap=cmap,
        vmin=min_z,
        vmax=max_z,
    )
    cb1 = plt.colorbar(cont, ax=ax_contour)
    cb1.ax.tick_params(labelsize=legend_fontsize - 2)

    ax_contour.set_xlabel(var_name[i], fontsize=legend_fontsize)
    ax_contour.set_ylabel(var_name[j], fontsize=legend_fontsize)
    ax_contour.tick_params(labelsize=legend_fontsize - 2)
    if aspect_equal:
        ax_contour.set_aspect("equal")

    # --- 3D surface ---
    surf = ax_surface.plot_surface(
        X_grid,
        Y_grid,
        Z_pred,
        cmap=cmap,
        vmin=min_z,
        vmax=max_z,
        linewidth=0,
        antialiased=True,
        alpha=0.8,
    )
    cb2 = fig.colorbar(surf, ax=ax_surface, shrink=0.7, pad=0.1)
    cb2.ax.tick_params(labelsize=legend_fontsize - 2)

    ax_surface.set_xlabel(var_name[i], fontsize=legend_fontsize)
    ax_surface.set_ylabel(var_name[j], fontsize=legend_fontsize)
    ax_surface.set_zlabel("f(x)", fontsize=legend_fontsize)
    ax_surface.tick_params(labelsize=legend_fontsize - 2)

    # --- Optionally plot points ---
    if plot_points and X_points is not None:
        # Build + predict each point individually, using the same i/j from the row
        # and the use_min/use_max logic for leftover dims. Store these predicted values
        # as "z_pred_for_point".
        z_pred_for_point = []
        for row_idx in range(X_points.shape[0]):
            single_p = np.zeros(n_dims)
            single_p[i] = X_points[row_idx, i]
            single_p[j] = X_points[row_idx, j]
            for dim_idx in range(n_dims):
                if dim_idx not in (i, j):
                    single_p[dim_idx] = hidden_value(dim_idx)
            val = model.predict(single_p.reshape(1, -1))
            val = np.atleast_1d(val)  # ensure at least 1D
            if isinstance(val, dict):
                val = val.get("mean", list(val.values())[0])
            elif isinstance(val, tuple):
                val = val[0]
            z_pred_for_point.append(val[0] if hasattr(val, "__len__") else val)
        z_pred_for_point = np.array(z_pred_for_point)

        # Use the ground-truth y_points directly:
        z_actual = np.array(y_points).flatten()  # ensures a 1D shape

        on_mask = np.isclose(z_actual, z_pred_for_point, atol=atol)
        below_mask = z_actual - atol / 2.0 < z_pred_for_point
        above_mask = z_actual + atol / 2.0 > z_pred_for_point
        num_correct = np.count_nonzero(on_mask)

        # 2D contour scatter
        ax_contour.scatter(
            X_points[below_mask, i],
            X_points[below_mask, j],
            c=point_color_below,
            edgecolor="black",
            s=points_size,
            alpha=0.9,
            zorder=5,
        )
        ax_contour.scatter(
            X_points[above_mask, i],
            X_points[above_mask, j],
            c=point_color_above,
            edgecolor="black",
            s=points_size,
            alpha=0.9,
            zorder=5,
        )
        ax_contour.scatter(
            X_points[on_mask, i],
            X_points[on_mask, j],
            c=points_color,
            edgecolor="black",
            s=points_size,
            alpha=0.9,
            zorder=5,
        )
        # 3D plot scatter
        ax_surface.scatter(
            X_points[below_mask, i],
            X_points[below_mask, j],
            z_actual[below_mask],
            c=point_color_below,
            edgecolor="black",
            s=points_size,
            alpha=0.9,
            zorder=10,  # Ensure points are rendered on top
        )
        ax_surface.scatter(
            X_points[above_mask, i],
            X_points[above_mask, j],
            z_actual[above_mask],
            c=point_color_above,
            edgecolor="black",
            s=points_size,
            alpha=0.9,
            zorder=10,  # Ensure points are rendered on top
        )
        ax_surface.scatter(
            X_points[on_mask, i],
            X_points[on_mask, j],
            z_actual[on_mask],
            c=points_color,
            edgecolor="black",
            s=points_size,
            alpha=0.9,
            zorder=10,  # Ensure points are rendered on top
        )

    # --- Optionally set aspect in 3D ---
    if aspect_equal:
        x_range = upper[i] - lower[i]
        y_range = upper[j] - lower[j]
        z_range = max_z - min_z if max_z > min_z else 1
        scale_z = (x_range + y_range) / (2.0 * z_range) if z_range else 1
        ax_surface.set_box_aspect([1, (y_range / x_range) if x_range else 1, scale_z])

    # --- Title, save, and show ---
    if title:
        updated_title = f"{title}  Correct Points: {num_correct}"
        fig.suptitle(updated_title, fontsize=legend_fontsize + 2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=dpi)

    if show:
        plt.show()

    return fig, (ax_contour, ax_surface)


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
    y_points=None,
    plot_points=True,
    points_color="white",
    points_size=30,
    point_color_below="blue",
    point_color_above="red",
    atol=1e-6,
) -> None:
    """Plot model surfaces for multiple combinations of input variables.

    This function generates contour and 3D surface plots for all specified
    combinations of input variables, avoiding redundancies and meaningless combinations.

    Args:
        model (object): A fitted model with a predict method.
        X (Optional[np.ndarray]): Array of training points. If provided, used to derive bounds and dimension count.
        lower (Optional[Union[np.ndarray, List[float]]]): Array-like lower bounds for each dimension. If None, derived from X.
        upper (Optional[Union[np.ndarray, List[float]]]): Array-like upper bounds for each dimension. If None, derived from X.
        x_vars (Optional[List[int]]): List of indices for x-axis variables. Defaults to all if None or empty.
        y_vars (Optional[List[int]]): List of indices for y-axis variables. Defaults to all if None or empty.
        min_z (Optional[float]): Min value for color scale. If None, auto-calculated.
        max_z (Optional[float]): Max value for color scale. If None, auto-calculated.
        var_type (Optional[List[str]]): List of variable types. If None, assumed numeric.
        var_name (Optional[List[str]]): List of variable names. If None, named x0, x1, ...
        show (bool): Whether to display the plots. Defaults to True.
        save_dir (Optional[str]): Directory for saving plots. If None, not saved.
        n_grid (int): Number of grid points along each axis. Defaults to 50.
        contour_levels (int): Number of contour levels. Defaults to 10.
        dpi (int): DPI for saving figures. Defaults to 200.
        title_prefix (str): Prefix string for plot titles.
        figsize (Tuple[float, float]): Figure size (width, height). Defaults to (12, 6).
        use_min (bool): Use lower bounds for non-plotted dimensions. Defaults to False.
        use_max (bool): Use upper bounds for non-plotted dimensions. Defaults to False.
        margin (float): Fraction of range added as margin to bounds when derived from X. Defaults to 0.1.
        aspect_equal (bool): Whether to set equal aspect ratio. Defaults to False.
        legend_fontsize (int): Font size for labels and legends. Defaults to 12.
        cmap (str): Colormap for the plots. Defaults to "viridis".
        X_points (Optional[np.ndarray]): Original data points to plot.
        y_points (Optional[np.ndarray]): Original target values to plot.
        plot_points (bool): Whether to plot X_points. Defaults to True.
        points_color (str): Color for data points. Defaults to "white".
        points_size (int): Marker size for data points. Defaults to 30.
        point_color_below (str): Color if actual z < predicted z. Defaults to "lightgrey".
        point_color_above (str): Color if actual z >= predicted z. Defaults to "white".
        atol (float): Absolute tolerance for comparing actual and predicted z-values. Defaults to 1e-6.

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
                y_points=y_points,  # Pass original target values
                plot_points=plot_points,
                points_color=points_color,
                points_size=points_size,
                point_color_below=point_color_below,
                point_color_above=point_color_above,
                atol=atol,
            )

    return None


def create_contour_plot(data, x_col, y_col, z_col, facet_col=None, aspect=1, as_table=True, figsize=(3, 3), levels=5, cmap="viridis") -> None:
    """
    Creates contour plots similar to R's contourplot function using matplotlib.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column to use for the x-axis.
        y_col (str): The name of the column to use for the y-axis.
        z_col (str): The name of the column to use for the z-axis (contour values).
        facet_col (str, optional): The name of the column to use for faceting (creating subplots). Defaults to None.
        aspect (float, optional): The aspect ratio of the plot. Defaults to 1.
        as_table (bool, optional): Whether to arrange facets as a table. Defaults to True.
        figsize (tuple, optional): The size of the figure. Defaults to (3, 3).
        levels (int, optional): The number of contour levels. Defaults to 5.
        cmap (str, optional): The colormap to use. Defaults to "viridis".

    Returns:
        None: Displays the contour plot(s).

    Raises:
        ValueError: If the specified columns are not found in the DataFrame.

    Examples:
        >>> from spotpython.plot.contour import create_contour_plot
            import numpy as np
            import pandas as pd
            # Create a grid of x and y values
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            x_grid, y_grid = np.meshgrid(x, y)
            # Calculate z = x^2 + y^2
            z = x_grid**2 + y_grid**2
            # Flatten the grid and create a DataFrame
            data = pd.DataFrame({
                'x': x_grid.flatten(),
                'y': y_grid.flatten(),
                'z': z.flatten()
            })
            # Create the contour plot
            create_contour_plot(data, 'x', 'y', 'z', facet_col=None)
        >>> # Create a contour plot with faceting
            from spotpython.plot.contour import create_contour_plot
            import numpy as np
            import pandas as pd
            # Create a grid of x and y values
            x = np.linspace(-5, 5, 50)
            y = np.linspace(-5, 5, 50)
            x_grid, y_grid = np.meshgrid(x, y)
            # Calculate z = x^2 + y^2 for two different facets
            z1 = x_grid**2 + y_grid**2
            z2 = (x_grid - 2)**2 + (y_grid - 2)**2
            # Flatten the grids and create a DataFrame
            data = pd.DataFrame({
                'x': np.tile(x, len(y) * 2),  # Repeat x values for both facets
                'y': np.repeat(y, len(x) * 2),  # Repeat y values for both facets
                'z': np.concatenate([z1.flatten(), z2.flatten()]),  # Combine z values for both facets
                'facet': ['Facet A'] * len(z1.flatten()) + ['Facet B'] * len(z2.flatten())  # Create facet column
            })
            # Create the contour plot with facets
            create_contour_plot(data, 'x', 'y', 'z', facet_col='facet')
    """

    if facet_col:
        facet_values = data[facet_col].unique()
        num_facets = len(facet_values)

        # Determine subplot layout
        if as_table:
            num_cols = int(np.ceil(np.sqrt(num_facets)))
            num_rows = int(np.ceil(num_facets / num_cols))
        else:
            num_cols = num_facets
            num_rows = 1

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0] * num_cols, figsize[1] * num_rows))
        axes = np.array(axes).flatten()  # Flatten the axes array for easy indexing

        for i, facet_value in enumerate(facet_values):
            ax = axes[i]
            facet_data = data[data[facet_col] == facet_value]

            # Create grid for contour plot
            x = np.unique(facet_data[x_col])
            y = np.unique(facet_data[y_col])
            X, Y = np.meshgrid(x, y)
            Z = facet_data.pivot_table(index=y_col, columns=x_col, values=z_col).values

            # Plot contour
            contour = ax.contour(X, Y, Z, levels=levels, cmap=cmap)  # Adjust levels and cmap as needed
            ax.clabel(contour, inline=True, fontsize=8)

            # Set labels and title
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{facet_col} = {np.round(facet_value, 2)}")
            ax.set_aspect(aspect)

        # Remove empty subplots
        for i in range(num_facets, len(axes)):
            fig.delaxes(axes[i])

        fig.tight_layout()
        plt.show()

    else:
        # Create grid for contour plot
        x = np.unique(data[x_col])
        y = np.unique(data[y_col])
        X, Y = np.meshgrid(x, y)
        Z = data.pivot_table(index=y_col, columns=x_col, values=z_col).values

        # Plot contour
        fig, ax = plt.subplots(figsize=figsize)
        contour = ax.contour(X, Y, Z, levels=levels, cmap="viridis")  # Adjust levels and cmap as needed
        ax.clabel(contour, inline=True, fontsize=8)

        # Set labels and title
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Contour Plot of {z_col}")
        ax.set_aspect(aspect)

        plt.show()


def mo_generate_plot_grid(variables, resolutions, functions) -> pd.DataFrame:
    """
    Generate a grid of input variables and apply objective functions.

    Args:
        variables (dict): A dictionary where keys are variable names (e.g., "time", "temperature")
                          and values are tuples of (min_value, max_value).
        resolutions (dict): A dictionary where keys are variable names and values are the number of points.
        functions (dict): A dictionary where keys are function names and values are callable functions.

    Returns:
        pd.DataFrame: A DataFrame containing the grid and the results of the objective functions.
    """
    # Create a meshgrid for all variables
    grids = [np.linspace(variables[var][0], variables[var][1], resolutions[var]) for var in variables]
    grid = np.array(np.meshgrid(*grids)).T.reshape(-1, len(variables))

    # Create a DataFrame for the grid
    plot_grid = pd.DataFrame(grid, columns=variables.keys())

    # Apply each function to the grid
    for func_name, func in functions.items():
        plot_grid[func_name] = plot_grid.apply(lambda row: func(row.values), axis=1)

    return plot_grid


def contour_plot(
    data,
    x_col,
    y_col,
    z_col,
    facet_col=None,
    aspect=1,
    as_table=True,
    figsize=(4, 4),
    levels=10,
    cmap="viridis",
    highlight_point=None,
    highlight_color="red",
    highlight_size=50,
    highlight_label=None,
    highlight_legend_loc="upper right",
    highlight_legend_fontsize=8,
) -> None:
    """
    Creates contour plots (single or faceted) using matplotlib.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column to use for the x-axis.
        y_col (str): The name of the column to use for the y-axis.
        z_col (str): The name of the column to use for the z-axis (contour values).
        facet_col (str, optional): The name of the column to use for faceting (creating subplots). Defaults to None.
        aspect (float, optional): The aspect ratio of the plot. Defaults to 1.
        as_table (bool, optional): Whether to arrange facets as a table. Defaults to True.
        figsize (tuple, optional): The size of the figure. Defaults to (4, 4).
        levels (int, optional): The number of contour levels. Defaults to 5.
        cmap (str, optional): The colormap to use. Defaults to "viridis".
        highlight_point (np.array, optional): A 1-dimensional array specifying a single point [x, y] to highlight. Defaults to None.
        highlight_color (str, optional): Color for the highlighted point. Defaults to "red".
        highlight_size (int, optional): Size of the highlighted point. Defaults to 50.
        highlight_label (str, optional): Label for the highlighted point. Defaults to "Highlighted Point".
        highlight_legend_loc (str, optional): Location for the legend. Defaults to "upper right".
        highlight_legend_fontsize (int, optional): Font size for the legend. Defaults to 8.

    Returns:
        None: Displays the contour plot(s).
    """
    if facet_col:
        facet_values = data[facet_col].unique()
        num_facets = len(facet_values)

        # Determine subplot layout
        if as_table:
            num_cols = int(np.ceil(np.sqrt(num_facets)))
            num_rows = int(np.ceil(num_facets / num_cols))
        else:
            num_cols = num_facets
            num_rows = 1

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0] * num_cols, figsize[1] * num_rows))
        axes = np.array(axes).flatten()  # Flatten the axes array for easy indexing

        for i, facet_value in enumerate(facet_values):
            ax = axes[i]
            facet_data = data[data[facet_col] == facet_value]

            # Create grid for contour plot
            x = np.unique(facet_data[x_col])
            y = np.unique(facet_data[y_col])
            X, Y = np.meshgrid(x, y)
            Z = facet_data.pivot_table(index=y_col, columns=x_col, values=z_col).values

            # Plot contour
            contour = ax.contour(X, Y, Z, levels=levels, cmap=cmap)
            ax.clabel(contour, inline=True, fontsize=8)

            # Highlight the specified point
            if highlight_point is not None:
                ax.scatter(highlight_point[0], highlight_point[1], color=highlight_color, s=highlight_size, label=highlight_label, zorder=10)
                if highlight_label:
                    ax.legend(loc=highlight_legend_loc, fontsize=highlight_legend_fontsize)

            # Set labels and title
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{facet_col} = {np.round(facet_value, 2)}")
            ax.set_aspect(aspect)

        # Remove empty subplots
        for i in range(num_facets, len(axes)):
            fig.delaxes(axes[i])

        fig.tight_layout()
        plt.show()

    else:
        # Create grid for contour plot
        x = np.unique(data[x_col])
        y = np.unique(data[y_col])
        X, Y = np.meshgrid(x, y)
        Z = data.pivot_table(index=y_col, columns=x_col, values=z_col).values

        # Plot contour
        fig, ax = plt.subplots(figsize=figsize)
        contour = ax.contour(X, Y, Z, levels=levels, cmap=cmap)
        ax.clabel(contour, inline=True, fontsize=8)

        # Highlight the specified point
        if highlight_point is not None:
            ax.scatter(highlight_point[0], highlight_point[1], color=highlight_color, s=highlight_size, label=highlight_label, zorder=10)
            if highlight_label:
                ax.legend(loc=highlight_legend_loc, fontsize=highlight_legend_fontsize)

        # Set labels and title
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Contour Plot of {z_col}")
        ax.set_aspect(aspect)

        plt.show()


def contourf_plot(
    data,
    x_col,
    y_col,
    z_col,
    facet_col=None,
    aspect=1,
    as_table=True,
    figsize=(4, 4),
    levels=10,
    cmap="viridis",
    show_contour_lines=True,
    contour_line_color="black",
    contour_line_width=0.5,
    colorbar_orientation="vertical",
    wspace=0.4,
    hspace=0.4,
    highlight_point=None,  # New argument to specify a single point to highlight
    highlight_color="red",  # Color for the highlighted point
    highlight_size=50,  # Size of the highlighted point
    highlight_label=None,  # Label for the highlighted point
    highlight_legend_loc="upper right",  # Legend location
    highlight_legend_fontsize=8,  # Font size for the legend
) -> None:
    """
    Creates filled contour plots (single or faceted) using matplotlib.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column to use for the x-axis.
        y_col (str): The name of the column to use for the y-axis.
        z_col (str): The name of the column to use for the z-axis (contour values).
        facet_col (str, optional): The name of the column to use for faceting (creating subplots). Defaults to None.
        aspect (float, optional): The aspect ratio of the plot. Defaults to 1.
        as_table (bool, optional): Whether to arrange facets as a table. Defaults to True.
        figsize (tuple, optional): The size of the figure. Defaults to (4, 4).
        levels (int, optional): The number of contour levels. Defaults to 10.
        cmap (str, optional): The colormap to use. Defaults to "viridis".
        show_contour_lines (bool, optional): Whether to overlay contour lines on the filled plot. Defaults to False.
        contour_line_color (str, optional): Color of the contour lines. Defaults to "black".
        contour_line_width (float, optional): Width of the contour lines. Defaults to 0.5.
        colorbar_orientation (str, optional): Orientation of the colorbar ("vertical" or "horizontal"). Defaults to "vertical".
        wspace (float, optional): Horizontal spacing between subplots. Defaults to 0.4.
        hspace (float, optional): Vertical spacing between subplots. Defaults to 0.4.
        highlight_point (np.array, optional): A 1-dimensional array specifying a single point [x, y] to highlight. Defaults to None.
        highlight_color (str, optional): Color for the highlighted point. Defaults to "red".
        highlight_size (int, optional): Size of the highlighted point. Defaults to 50.
        highlight_label (str, optional): Label for the highlighted point. Defaults to None.
        highlight_legend_loc (str, optional): Location for the legend. Defaults to "upper right".
        highlight_legend_fontsize (int, optional): Font size for the legend. Defaults to 8.

    Returns:
        None: Displays the filled contour plot(s).
    """
    if facet_col:
        facet_values = data[facet_col].unique()
        num_facets = len(facet_values)

        # Determine subplot layout
        if as_table:
            num_cols = int(np.ceil(np.sqrt(num_facets)))
            num_rows = int(np.ceil(num_facets / num_cols))
        else:
            num_cols = num_facets
            num_rows = 1

        # Create figure with gridspec for colorbar placement
        if colorbar_orientation == "vertical":
            fig = plt.figure(figsize=(figsize[0] * num_cols, figsize[1] * num_rows))
            spec = gridspec.GridSpec(num_rows, num_cols + 1, width_ratios=[1] * num_cols + [0.05], wspace=wspace, hspace=hspace)
        else:  # Horizontal colorbar
            fig = plt.figure(figsize=(figsize[0] * num_cols, figsize[1] * num_rows + 1))
            spec = gridspec.GridSpec(num_rows + 1, num_cols, height_ratios=[1] * num_rows + [0.05], wspace=wspace, hspace=hspace)

        axes = []
        for row in range(num_rows):
            for col in range(num_cols):
                if row * num_cols + col < num_facets:
                    axes.append(fig.add_subplot(spec[row, col]))

        for i, facet_value in enumerate(facet_values):
            ax = axes[i]
            facet_data = data[data[facet_col] == facet_value]

            # Create grid for contour plot
            x = np.unique(facet_data[x_col])
            y = np.unique(facet_data[y_col])
            X, Y = np.meshgrid(x, y)
            Z = facet_data.pivot_table(index=y_col, columns=x_col, values=z_col).values

            # Plot filled contour
            contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)

            # Optionally overlay contour lines
            if show_contour_lines:
                contour_lines = ax.contour(X, Y, Z, levels=levels, colors=contour_line_color, linewidths=contour_line_width)
                ax.clabel(contour_lines, inline=True, fontsize=8)

            # Highlight the specified point
            if highlight_point is not None:
                ax.scatter(highlight_point[0], highlight_point[1], color=highlight_color, s=highlight_size, label=highlight_label, zorder=10)
                if highlight_label:
                    ax.legend(loc=highlight_legend_loc, fontsize=highlight_legend_fontsize)

            # Set labels and title
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{facet_col} = {np.round(facet_value, 2)}")
            ax.set_aspect(aspect)

        # Add colorbar
        if colorbar_orientation == "vertical":
            cbar_ax = fig.add_subplot(spec[:, -1])  # Last column for vertical colorbar
        else:
            cbar_ax = fig.add_subplot(spec[-1, :])  # Last row for horizontal colorbar
        fig.colorbar(contour, cax=cbar_ax, orientation=colorbar_orientation, label=z_col)

        plt.show()

    else:
        # Create grid for contour plot
        x = np.unique(data[x_col])
        y = np.unique(data[y_col])
        X, Y = np.meshgrid(x, y)
        Z = data.pivot_table(index=y_col, columns=x_col, values=z_col).values

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot filled contour
        contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)

        # Optionally overlay contour lines
        if show_contour_lines:
            contour_lines = ax.contour(X, Y, Z, levels=levels, colors=contour_line_color, linewidths=contour_line_width)
            ax.clabel(contour_lines, inline=True, fontsize=8)

        # Highlight the specified point
        if highlight_point is not None:
            ax.scatter(highlight_point[0], highlight_point[1], color=highlight_color, s=highlight_size, label=highlight_label, zorder=10)
            if highlight_label:
                ax.legend(loc=highlight_legend_loc, fontsize=highlight_legend_fontsize)

        # Set labels and title
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Filled Contour Plot of {z_col}")
        ax.set_aspect(aspect)

        # Add colorbar
        if colorbar_orientation == "vertical":
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position for vertical colorbar
        else:
            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # Position for horizontal colorbar
        fig.colorbar(contour, cax=cbar_ax, orientation=colorbar_orientation, label=z_col)

        plt.show()
