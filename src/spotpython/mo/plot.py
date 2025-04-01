import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spotpython.mo.pareto import is_pareto_efficient


def plot_mo(
    target_names: list,
    combinations: list,
    pareto: str,
    y_rf: np.ndarray = None,
    pareto_front: bool = False,
    y_best: np.ndarray = None,
    y_add: np.ndarray = None,
    y_add2: np.ndarray = None,
    y_add_color="blue",
    y_add2_color="green",
    title: str = "",
    y_orig: np.ndarray = None,
    pareto_front_orig: bool = False,
    pareto_label: bool = False,
    y_rf_color="blue",
    y_best_color="red",
    x_axis_transformation: str = "id",
    y_axis_transformation: str = "id",
    y_best_label="Best",
    y_add_label="Add",
    y_add2_label="Add2",
    filename: str = None,
) -> None:
    """
    Generates scatter plots for each combination of two targets from a multi-output prediction while highlighting Pareto optimal points.

    Args:
        y_rf (np.ndarray): The predicted target values with shape (n_samples, n_targets).
        target_names (list): A list of target names corresponding to the columns of y_rf.
        combinations (list): A list of tuples, where each tuple contains the indices of the target combinations to plot.
        pareto (str): Specifies whether to compute Pareto front based on 'min' or 'max' criterion.
        pareto_front (bool): If True, connect Pareto optimal points with a red line for y_rf.
        y_best (np.ndarray, optional): A NumPy array representing the best point to highlight in red. Defaults to None.
        y_add (np.ndarray, optional): A NumPy array representing the additional points to highlight in blue. Defaults to None.
        y_add2 (np.ndarray, optional): A NumPy array representing the additional points to highlight in green. Defaults to None.
        y_add_color (str): The color of the additional points. Defaults to "blue".
        y_add2_color (str): The color of the additional points. Defaults to "green".
        y_best_label (str): The label for the best point. Defaults to "Best".
        y_add_label (str): The label for the additional points. Defaults to "Add".
        y_add2_label (str): The label for the additional points. Defaults to "Add2".
        title (str): The title of the plot. Defaults to "" (empty string).
        y_orig (np.ndarray, optional): The original target values with shape (n_samples, n_targets). Defaults to None.
        pareto_front_orig (bool): If True, connect Pareto optimal points with a light blue line for y_orig. Defaults to False.
        pareto_label (bool): If True, label Pareto points with their index. Defaults to False.
        y_rf_color (str): The color of the predicted points. Defaults to "blue".
        y_best_color (str): The color of the best point. Defaults to "red".
        x_axis_transformation (str): Transformation for the x-axis. Options are "id" (linear), "log" (logarithmic), and "loglog" (log-log). Defaults to "id".
        y_axis_transformation (str): Transformation for the y-axis. Options are "id" (linear), "log" (logarithmic), and "loglog" (log-log). Defaults to "id".
        filename (str, optional):
            If provided, saves the plot to the specified file.  Supports "pdf" and "png" formats. Defaults to None.

    Returns:
        None: Displays or saves the plot.

    Examples:
        >>> from spotpython.mo.plot import plot_mo
        >>> import numpy as np
        >>> target_names = ["Target 1", "Target 2"]
        >>> combinations = [(0, 1)]
        >>> pareto = "min"
        >>> y_rf = np.random.rand(100, 2)
        >>> y_orig = np.random.rand(100, 2)
        >>> plot_mo(target_names, combinations, pareto, y_rf=y_rf, y_orig=y_orig, filename="plot.png")
    """
    # Convert y_rf to numpy array if it's a pandas DataFrame
    if isinstance(y_rf, pd.DataFrame):
        y_rf = y_rf.values

    # Convert y_orig to numpy array if it's a pandas DataFrame
    if isinstance(y_orig, pd.DataFrame):
        y_orig = y_orig.values

    for i, j in combinations:
        plt.figure()
        s = 50  # Base size for points
        pareto_size = s  # Size for Pareto points
        if pareto_label:
            pareto_size = s * 4  # Increase the size for Pareto points
        a = 0.4

        # Plot original data if provided
        if y_orig is not None:
            minimize = pareto == "min"
            pareto_mask_orig = is_pareto_efficient(y_orig[:, [i, j]], minimize)
            plt.scatter(y_orig[:, i], y_orig[:, j], edgecolor="w", c="gray", s=s, marker="o", alpha=a, label="Non-Pareto Points")
            plt.scatter(y_orig[pareto_mask_orig, i], y_orig[pareto_mask_orig, j], edgecolor="k", c="gray", s=pareto_size, marker="o", alpha=a, label="Pareto Points")
            if pareto_label:
                for idx in np.where(pareto_mask_orig)[0]:
                    plt.text(y_orig[idx, i], y_orig[idx, j], str(idx), color="black", fontsize=8, ha="center", va="center")
            if pareto_front_orig:
                sorted_indices_orig = np.argsort(y_orig[pareto_mask_orig, i])
                plt.plot(y_orig[pareto_mask_orig, i][sorted_indices_orig], y_orig[pareto_mask_orig, j][sorted_indices_orig], "k-", alpha=a, label="Pareto Front")

        if y_rf is not None:
            minimize = pareto == "min"
            pareto_mask = is_pareto_efficient(y_rf[:, [i, j]], minimize)
            plt.scatter(y_rf[:, i], y_rf[:, j], edgecolor="w", c=y_rf_color, s=s, marker="^", alpha=a, label="Predicted Points")
            plt.scatter(y_rf[pareto_mask, i], y_rf[pareto_mask, j], edgecolor="k", c=y_rf_color, s=pareto_size, marker="s", alpha=a, label="Predicted Pareto")
            if pareto_label:
                for idx in np.where(pareto_mask)[0]:
                    plt.text(y_rf[idx, i], y_rf[idx, j], str(idx), color="black", fontsize=8, ha="center", va="center")
            if pareto_front:
                sorted_indices = np.argsort(y_rf[pareto_mask, i])
                plt.plot(
                    y_rf[pareto_mask, i][sorted_indices],
                    y_rf[pareto_mask, j][sorted_indices],
                    linestyle="-",
                    color=y_rf_color,
                    alpha=a,
                    label="Predicted Pareto Front",
                )

        if y_best is not None:
            plt.scatter(y_best[:, i], y_best[:, j], edgecolor="k", c=y_best_color, s=s, marker="D", alpha=1, label=y_best_label)
        if y_add is not None:
            plt.scatter(y_add[:, i], y_add[:, j], edgecolor="k", c=y_add_color, s=s, marker="D", alpha=1, label=y_add_label)
        if y_add2 is not None:
            plt.scatter(y_add2[:, i], y_add2[:, j], edgecolor="k", c=y_add2_color, s=s, marker="D", alpha=1, label=y_add2_label)

        # Apply axis transformations
        if x_axis_transformation == "log":
            plt.xscale("log")
        if y_axis_transformation == "log":
            plt.yscale("log")
        if x_axis_transformation == "loglog" or y_axis_transformation == "loglog":
            plt.xscale("log")
            plt.yscale("log")

        plt.xlabel(target_names[i])
        plt.ylabel(target_names[j])
        plt.grid()
        plt.title(title)
        # Move the legend outside the plot
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # Save or show the plot
        if filename:
            if filename.endswith(".pdf") or filename.endswith(".png"):
                plt.savefig(filename, format=filename.split(".")[-1], bbox_inches="tight")
            else:
                raise ValueError("Filename must have a valid suffix: '.pdf' or '.png'.")
        else:
            plt.show()
