from spotpython.data.friedman import FriedmanDriftDataset
import matplotlib.pyplot as plt


def plot_friedman_drift_data(n_samples, seed, change_point1, change_point2, constant=True, show=True, filename=None) -> None:
    """Plot the Friedman dataset with drifts at change_point1 and change_point2.

    Args:
        n_samples (int):
            Number of samples to generate.
        seed (int):
            Seed for the random number generator.
        change_point1 (int):
            Index of the first drift point.
        change_point2 (int):
            Index of the second drift point.
        constant (bool, optional):
            If True, the drifts are constant. Defaults to True.
        filename (str, optional):
            Name of the file to save the plot. Defaults to None.

    Returns:
        None

    Examples:
        >>> from spotpython.plot.ts import plot_friedman_drift_data
        >>> plot_friedman_drift_data(n_samples=100, seed=42, change_point1=50, change_point2=75, constant=False)
        >>> plot_friedman_drift_data(n_samples=100, seed=42, change_point1=50, change_point2=75, constant=True)
    """
    data_generator = FriedmanDriftDataset(n_samples=n_samples, seed=seed, change_point1=change_point1, change_point2=change_point2, constant=constant)
    data = [data for data in data_generator]
    indices = [i for _, _, i in data]
    values = {f"x{i}": [] for i in range(6)}
    values["y"] = []
    for x, y, _ in data:
        for i in range(6):
            values[f"x{i}"].append(x[i])
        values["y"].append(y)

    plt.figure(figsize=(10, 6))
    for label, series in values.items():
        plt.plot(indices, series, label=label)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.axvline(x=change_point1, color="k", linestyle="--", label="Drift Point 1")
    plt.axvline(x=change_point2, color="r", linestyle="--", label="Drift Point 2")
    plt.legend()
    plt.grid(True)
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
