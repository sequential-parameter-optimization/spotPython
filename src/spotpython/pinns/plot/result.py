import torch
import matplotlib.pyplot as plt
from typing import Optional, Union, List
import numpy as np


def plot_result(
    x: Union[torch.Tensor, List[float], "np.ndarray"],
    y: Union[torch.Tensor, List[float], "np.ndarray"],
    x_data: Union[torch.Tensor, List[float], "np.ndarray"],
    y_data: Union[torch.Tensor, List[float], "np.ndarray"],
    yh: Union[torch.Tensor, List[float], "np.ndarray"],
    current_step: int,
    xp: Optional[Union[torch.Tensor, List[float], "np.ndarray"]] = None,
    figure_size: tuple = (8, 4),
    xlims: Optional[tuple] = (-1.25, 31.05),
    ylims: Optional[tuple] = (-0.65, 2.25),
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Plots the results of a PINN training, comparing predictions with exact solutions.

    Displays the neural network's prediction, the exact solution, training data points,
    and optionally, collocation points.

    Args:
        x (Union[torch.Tensor, List[float], "np.ndarray"]):
            The x-coordinates for the continuous plots (e.g., time points).
        y (Union[torch.Tensor, List[float], "np.ndarray"]):
            The y-coordinates of the exact solution corresponding to `x`.
        x_data (Union[torch.Tensor, List[float], "np.ndarray"]):
            The x-coordinates of the training data points.
        y_data (Union[torch.Tensor, List[float], "np.ndarray"]):
            The y-coordinates of the training data points.
        yh (Union[torch.Tensor, List[float], "np.ndarray"]):
            The y-coordinates of the neural network's prediction corresponding to `x`.
        current_step (int):
            The current training step or epoch number to display on the plot.
        xp (Optional[Union[torch.Tensor, List[float], "np.ndarray"]], optional):
            The x-coordinates of the collocation points. If None, these are not plotted.
            Defaults to None.
        figure_size (tuple, optional):
            Size of the matplotlib figure. Defaults to (8, 4).
        xlims (Optional[tuple], optional):
            Tuple defining the x-axis limits. If None, matplotlib's default is used.
            Defaults to (-1.25, 31.05).
        ylims (Optional[tuple], optional):
            Tuple defining the y-axis limits. If None, matplotlib's default is used.
            Defaults to (-0.65, 2.25).
        show_plot (bool, optional):
            Whether to display the plot using `plt.show()`. Defaults to True.
        save_path (Optional[str], optional):
            If provided, the path to save the figure to. If None, the figure is not saved.
            Defaults to None.

    Examples:
        >>> from spotpython.pinns.plot.result import plot_result
        >>> import torch
        >>> import numpy as np
        >>> # Generate some dummy data
        >>> x_plot = torch.linspace(0, 30, 100)
        >>> y_exact = torch.sin(x_plot / 5)
        >>> y_pred = torch.sin(x_plot / 5 + 0.1) # Slightly off prediction
        >>> x_train = torch.rand(10) * 30
        >>> y_train = torch.sin(x_train / 5)
        >>> collocation_points = torch.rand(50) * 30
        >>> current_training_step = 1000
        >>> # plot_result( # This would show a plot if run in an interactive environment
        ... #     x_plot, y_exact, x_train, y_train, y_pred,
        ... #     current_training_step, xp=collocation_points,
        ... #     show_plot=False, save_path="temp_plot.png"
        ... # )
        >>> # To avoid actual plotting in doctest, we'll just confirm it runs
        >>> try:
        ...     plot_result(
        ...         x_plot.numpy(), y_exact.numpy(), x_train.numpy(), y_train.numpy(), y_pred.numpy(),
        ...         current_training_step, xp=collocation_points.numpy(),
        ...         show_plot=False
        ...     )
        ... except Exception as e:
        ...     print(f"Plotting failed: {e}")

    Note:
        If using PyTorch tensors as input, they will be detached and moved to CPU
        before plotting. Consider converting to NumPy arrays beforehand if preferred.

    References:
        - Solving differential equations using physics informed deep learning: a hand-on tutorial with benchmark tests. Baty, Hubert and Baty, Leo. April 2023.
    """

    # Convert tensors to numpy arrays for plotting if they are tensors
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data

    x_np = to_numpy(x)
    y_np = to_numpy(y)
    x_data_np = to_numpy(x_data)
    y_data_np = to_numpy(y_data)
    yh_np = to_numpy(yh)
    if xp is not None:
        xp_np = to_numpy(xp)
    else:
        xp_np = None

    plt.figure(figsize=figure_size)
    plt.plot(x_np, yh_np, color="tab:red", linewidth=2, alpha=0.8, label="NN prediction")
    plt.plot(x_np, y_np, color="blue", linewidth=2, alpha=0.8, linestyle="--", label="Exact solution")
    plt.scatter(x_data_np, y_data_np, s=60, color="tab:red", alpha=0.4, label="Training data")

    if xp_np is not None:
        # Create y-values for collocation points at y=0 or a specified level
        # Original code used -0*torch.ones_like(xp), which is just zeros.
        xp_y_values = np.zeros_like(xp_np)
        plt.scatter(xp_np, xp_y_values, s=30, color="tab:green", alpha=0.4, label="Collocation points")

    legend_handle = plt.legend(loc=(0.67, 0.62), frameon=False, fontsize="large")
    plt.setp(legend_handle.get_texts(), color="k")

    if xlims:
        plt.xlim(xlims)
    if ylims:
        plt.ylim(ylims)

    plt.text(0.05, 0.95, f"Training step: {current_step}", fontsize="xx-large", color="k", transform=plt.gca().transAxes, ha="left", va="top")

    plt.ylabel("y", fontsize="xx-large")
    plt.xlabel("Time", fontsize="xx-large")
    plt.axis("on")
    plt.grid(True, linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure if not shown to free memory
