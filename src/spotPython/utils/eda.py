from tabulate import tabulate
from spotPython.hyperparameters.values import (
    get_default_values,
    get_bound_values,
    get_var_name,
    get_var_type,
    get_transform,
)
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
import numpy as np
from spotPython.utils.time import get_timestamp
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.colors as colors


def get_stars(input_list) -> list:
    """Converts a list of values to a list of stars, which can be used to
        visualize the importance of a variable.

    Args:
        input_list (list): A list of values.

    Returns:
        (list):
            A list of strings.

    Examples:
        >>> from spotPython.utils.eda import convert_list
        >>> get_stars([100, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        [***, '', '', '', '', '', '', '', '']
    """
    output_list = []
    for value in input_list:
        if value > 95:
            output_list.append("***")
        elif value > 50:
            output_list.append("**")
        elif value > 1:
            output_list.append("*")
        elif value > 0.1:
            output_list.append(".")
        else:
            output_list.append("")
    return output_list


def gen_design_table(fun_control: dict, spot: object = None, tablefmt="github") -> str:
    """Generates a table with the design variables and their bounds.
    Args:
        fun_control (dict):
            A dictionary with function design variables.
        spot (object):
            A spot object. Defaults to None.
    Returns:
        (str):
            a table with the design variables, their default values, and their bounds.
            If a spot object is provided,
            the table will also include the value and the importance of each hyperparameter.
            Use the `print` function to display the table.

    Examples:
        >>> from spotPython.utils.eda import gen_design_table
        >>> from spotPython.hyperparameters.values import get_default_values
        >>> fun_control = {
        ...     "x1": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x2": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x3": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x4": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x5": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x6": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x7": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x8": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x9": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x10": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ... }
    """
    default_values = get_default_values(fun_control)
    defaults = list(default_values.values())
    if spot is None:
        tab = tabulate(
            {
                "name": get_var_name(fun_control),
                "type": get_var_type(fun_control),
                "default": defaults,
                "lower": get_bound_values(fun_control, "lower", as_list=True),
                "upper": get_bound_values(fun_control, "upper", as_list=True),
                "transform": get_transform(fun_control),
            },
            headers="keys",
            tablefmt=tablefmt,
        )
    else:
        res = spot.print_results(print_screen=False, dict=fun_control)
        tuned = [item[1] for item in res]
        # imp = spot.print_importance(threshold=0.0, print_screen=False)
        # importance = [item[1] for item in imp]
        importance = spot.get_importance()
        stars = get_stars(importance)
        tab = tabulate(
            {
                "name": get_var_name(fun_control),
                "type": get_var_type(fun_control),
                "default": defaults,
                "lower": get_bound_values(fun_control, "lower", as_list=True),
                "upper": get_bound_values(fun_control, "upper", as_list=True),
                "tuned": tuned,
                "transform": get_transform(fun_control),
                "importance": importance,
                "stars": stars,
            },
            headers="keys",
            numalign="right",
            floatfmt=("", "", "", "", "", "", "", ".2f"),
            tablefmt=tablefmt,
        )
    return tab


def compare_two_tree_models(model1, model2, headers=["Parameter", "Default", "Spot"]):
    """Compares two tree models.
    Args:
        model1 (object):
            A tree model.
        model2 (object):
            A tree model.
        headers (list):
            A list with the headers of the table.

    Returns:
        (str):
            A table with the comparison of the two models.

    Examples:
        >>> from spotPython.utils.eda import compare_two_tree_models
        >>> from spotPython.hyperparameters.values import get_default_values
        >>> fun_control = {
        ...     "x1": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x2": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x3": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x4": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x5": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x6": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x7": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x8": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x9": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ...     "x10": {"type": "int", "default": 1, "lower": 1, "upper": 10},
        ... }
        >>> default_values = get_default_values(fun_control)
        >>> model1 = spot_tuner.get_model("rf", default_values)
        >>> model2 = spot_tuner.get_model("rf", default_values)
        >>> compare_two_tree_models(model1, model2)
    """
    keys = model1.summary.keys()
    values1 = model1.summary.values()
    values2 = model2.summary.values()
    tbl = []
    for key, value1, value2 in zip(keys, values1, values2):
        tbl.append([key, value1, value2])
    return tabulate(tbl, headers=headers, numalign="right", tablefmt="github")


def generate_config_id(config, hash=False, timestamp=False):
    """Generates a unique id for a configuration.

    Args:
        config (dict):
            A dictionary with the configuration.
        hash (bool):
            If True, the id is hashed.
        timestamp (bool):
            If True, the id is appended with a timestamp. Defaults to False.

    Returns:
        (str):
            A unique id for the configuration.

    Examples:
        >>> from spotPython.hyperparameters.values import get_one_config_from_X
        >>> X = spot_tuner.to_all_dim(spot_tuner.min_X.reshape(1,-1))
        >>> config = get_one_config_from_X(X, fun_control)
        >>> generate_config_id(config)
    """
    config_id = ""
    for key in config:
        # if config[key] is a number, round it to 4 digits after the decimal point
        if isinstance(config[key], float):
            config_id += str(round(config[key], 4)) + "_"
        else:
            config_id += str(config[key]) + "_"
    # hash the config_id to make it shorter and unique
    if hash:
        config_id = str(hash(config_id)) + "_"
    # remove () and , from the string
    config_id = config_id.replace("(", "")
    config_id = config_id.replace(")", "")
    config_id = config_id.replace(",", "")
    config_id = config_id.replace(" ", "")
    config_id = config_id.replace(":", "")
    if timestamp:
        config_id = get_timestamp(only_int=True) + "_" + config_id
    return config_id[:-1]


def filter_highly_correlated(df: pd.DataFrame, sorted: bool = True, threshold: float = 1 - 1e-5) -> pd.DataFrame:
    """
    Return a new DataFrame with only those columns that are highly correlated.

    Args:
        df (DataFrame): The input DataFrame.
        threshold (float): The correlation threshold.
        sorted (bool): If True, the columns are sorted by name.

    Returns:
        DataFrame: A new DataFrame with only highly correlated columns.

    Examples:
        >>> df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
            df = filter_highly_correlated(df, sorted=True, threshold=0.99)

    """
    corr_matrix = df.corr()
    # Find pairs of columns with correlation greater than threshold
    corr_pairs = corr_matrix.abs().unstack()
    corr_pairs = corr_pairs[corr_pairs < 1]  # Remove self-correlations
    high_corr = corr_pairs[corr_pairs > threshold]
    high_corr = high_corr[high_corr < 1]  # Remove self-correlations

    # Get the column names of highly correlated columns
    high_corr_cols = list(set([col[0] for col in high_corr.index]))

    # Create new DataFrame with only highly correlated columns
    new_df = df[high_corr_cols]
    # sort the columns by name
    if sorted:
        new_df = new_df.sort_index(axis=1)

    return new_df


def plot_sns_heatmap(
    df_heat,
    figsize=(16, 12),
    cmap="vlag",
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".5f",
    linewidths=0.5,
    annot_kws={"size": 8},
) -> None:
    """
    Plots a heatmap of the correlation matrix of the given DataFrame.

    Args:
        df_heat (pd.DataFrame): DataFrame containing the data to be plotted.
        figsize (tuple): Size of the figure to be plotted.
        cmap (str): Color map to be used for the heatmap.
        vmin (int): Minimum value for the color scale.
        vmax (int): Maximum value for the color scale.
        annot (bool): Whether to display annotations on the heatmap.
        fmt (str): Format string for annotations.
        linewidths (float): Width of lines separating cells in the heatmap.
        annot_kws (dict): Keyword arguments for annotations.

    Returns:
        (NoneType): None

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        >>> plot_heatmap(df)
    """
    plt.figure(figsize=figsize)
    matrix = np.triu(np.ones_like(df_heat.corr()))
    sns.heatmap(
        data=df_heat.corr(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
        linewidths=linewidths,
        annot_kws=annot_kws,
        mask=matrix,
    )
    plt.show()
    plt.gcf().clear()


def count_missing_data(df) -> pd.DataFrame:
    """
    Counts the number of missing values in each column of the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be counted.

    Returns:
        (pd.DataFrame): DataFrame containing the number of missing values in each column.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6], 'C': [7, 8, 9]})
        >>> count_missing_data(df)
           column_name  missing_count
        0           A              1
        1           B              1
    """
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ["column_name", "missing_count"]
    missing_df = missing_df.loc[missing_df["missing_count"] > 0]
    missing_df = missing_df.sort_values(by="missing_count")

    return missing_df


def plot_missing_data(
    df, relative=False, figsize=(7, 5), color="grey", xlabel="Missing Data", title="Missing Data"
) -> None:
    """
    Plots a horizontal bar chart of the number of missing values in each column of the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be plotted.
        relative (bool): Whether to plot relative values (percentage) or absolute values.
        figsize (tuple): Size of the figure to be plotted.
        color (str): Color of the bars in the bar chart.
        xlabel (str): Label for the x-axis.
        title (str): Title for the plot.

    Returns:
        (NoneType): None

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6], 'C': [7, 8, 9]})
        >>> plot_missing_data(df)
    """
    missing_df = count_missing_data(df)

    if relative:
        missing_df["missing_count"] = missing_df["missing_count"] / df.shape[0]
        xlabel = "Percentage of " + xlabel
        title = "Percentage of " + title

    ind = np.arange(missing_df.shape[0])
    _, ax = plt.subplots(figsize=figsize)
    _ = ax.barh(ind, missing_df.missing_count.values, color=color)
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.column_name.values, rotation="horizontal")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.vlines(1, 0, missing_df.shape[0])
    plt.vlines(0.97, 0, missing_df.shape[0])
    plt.vlines(0.5, 0, missing_df.shape[0])
    plt.show()


def visualize_activations(net, fun_control, batch_size=128, device="cpu", color="C0", columns=2) -> None:
    """Visualize the activations of a neural network.

    Args:
        net (object):
            A neural network.
        fun_control (dict):
            A dictionary with the function control parameters.
        color (str, optional):
            The color to use. Defaults to "C0".

    Notes:
        Code is based on: PyTorch Lightning TUTORIAL 2: ACTIVATION FUNCTIONS, Author: Phillip Lippe,  License: CC BY-SA.

    Examples:
        >>> from spotPython.hyperparameters.values import get_one_config_from_X
            X = spot_tuner.to_all_dim(spot_tuner.min_X.reshape(1,-1))
            config = get_one_config_from_X(X, fun_control)
            model = fun_control["core_model"](**config, _L_in=64, _L_out=11)
            visualize_activations(model, fun_control, color=f"C{0}")
    """
    activations = {}
    net.eval()
    dataset = fun_control["data_set"]
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # for batch in dataloader:
    #     inputs, targets = batch
    # small_loader = data.DataLoader(train_set, batch_size=1024)
    inputs, _ = next(iter(dataloader))
    with torch.no_grad():
        layer_index = 0
        inputs = inputs.to(device)
        inputs = inputs.view(inputs.size(0), -1)
        # We need to manually loop through the layers to save all activations
        # for layer_index, layer in enumerate(net.layers[:-1]):
        for layer_index, layer in enumerate(net.layers):
            inputs = layer(inputs)
            if isinstance(layer, nn.Linear):
                activations[layer_index] = inputs.view(-1).cpu().numpy()
    print(f"activations:{activations}")
    # Plotting
    n = len(activations)
    print(f"n:{n}")
    rows = n // columns + int(n % columns > 0)
    fig, ax = plt.subplots(rows, columns, figsize=(columns * 2.7, rows * 2.5))
    fig_index = 0
    for key in activations:
        key_ax = ax[fig_index // columns][fig_index % columns]
        sns.histplot(data=activations[key], bins=50, ax=key_ax, color=color, kde=True, stat="density")
        key_ax.set_title(f"Layer {key} - {net.layers[key].__class__.__name__}")
        fig_index += 1
    fig.suptitle(f"Activation distribution for activation function {net.hparams.act_fn}", fontsize=14)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()
    plt.close()


def visualize_weight_distributions(
    net, fun_control, batch_size=128, device="cpu", color="C0", xlabel=None, stat="count", use_kde=True, columns=2
) -> None:
    """Visualize the activations of a neural network.

    Args:
        net (object):
            A neural network.
        fun_control (dict):
            A dictionary with the function control parameters.
        color (str, optional):
            The color to use. Defaults to "C0".

    Notes:
        Code is based on: PyTorch Lightning TUTORIAL 2: ACTIVATION FUNCTIONS, Author: Phillip Lippe,  License: CC BY-SA.

    Examples:
        >>> from spotPython.hyperparameters.values import get_one_config_from_X
            X = spot_tuner.to_all_dim(spot_tuner.min_X.reshape(1,-1))
            config = get_one_config_from_X(X, fun_control)
            model = fun_control["core_model"](**config, _L_in=64, _L_out=11)
            visualize_activations(model, fun_control, color=f"C{0}")
    """
    weights = {}
    for name, param in net.named_parameters():
        if name.endswith(".bias"):
            continue
        key_name = f"Layer {name.split('.')[1]}"
        weights[key_name] = param.detach().view(-1).cpu().numpy()
    print(f"weights: {weights}")

    # Plotting
    n = len(weights)
    print(f"n:{n}")
    rows = n // columns + int(n % columns > 0)

    fig, ax = plt.subplots(rows, columns, figsize=(columns * 2.7, rows * 2.5))
    fig_index = 0
    for key in weights:
        key_ax = ax[fig_index // columns][fig_index % columns]
        sns.histplot(data=weights[key], bins=50, ax=key_ax, color=color, kde=True, stat="density")
        hidden_dim_str = (
            r"(%i $\to$ %i)" % (weights[key].shape[1], weights[key].shape[0]) if len(weights[key].shape) > 1 else ""
        )
        # key_ax.set_title(f"Layer {key} - {net.layers[key].__class__.__name__}")
        key_ax.set_title(f"{key} {hidden_dim_str}")
        fig_index += 1
    fig.suptitle(f"Weight distribution for activation function {net.hparams.act_fn}", fontsize=14)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()
    plt.close()


def visualize_gradients_distributions(
    net, fun_control, batch_size=128, device="cpu", color="C0", xlabel=None, stat="count", use_kde=True, columns=2
) -> None:
    """Visualize the activations of a neural network.

    Args:
        net (object):
            A neural network.
        fun_control (dict):
            A dictionary with the function control parameters.
        color (str, optional):
            The color to use. Defaults to "C0".

    Notes:
        Code is based on: PyTorch Lightning TUTORIAL 2: ACTIVATION FUNCTIONS, Author: Phillip Lippe,  License: CC BY-SA.

    Examples:
        >>> from spotPython.hyperparameters.values import get_one_config_from_X
            X = spot_tuner.to_all_dim(spot_tuner.min_X.reshape(1,-1))
            config = get_one_config_from_X(X, fun_control)
            model = fun_control["core_model"](**config, _L_in=64, _L_out=11)
            visualize_activations(model, fun_control, color=f"C{0}")
    """
    grads = {}
    net.eval()
    dataset = fun_control["data_set"]
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # for batch in dataloader:
    #     inputs, targets = batch
    # small_loader = data.DataLoader(train_set, batch_size=1024)
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)
    # Pass one batch through the network, and calculate the gradients for the weights
    net.zero_grad()
    preds = net(inputs)
    loss = F.mse_loss(preds, targets)
    # loss = F.cross_entropy(preds, labels)  # Same as nn.CrossEntropyLoss, but as a function instead of module
    loss.backward()
    # We limit our visualization to the weight parameters and exclude the bias to reduce the number of plots
    grads = {
        name: params.grad.view(-1).cpu().clone().numpy() for name, params in net.named_parameters() if "weight" in name
    }
    net.zero_grad()

    # Plotting
    n = len(grads)
    print(f"n:{n}")
    rows = n // columns + int(n % columns > 0)

    fig, ax = plt.subplots(rows, columns, figsize=(columns * 2.7, rows * 2.5))
    fig_index = 0
    for key in grads:
        key_ax = ax[fig_index // columns][fig_index % columns]
        sns.histplot(data=grads[key], bins=50, ax=key_ax, color=color, kde=True, stat="density")
        hidden_dim_str = (
            r"(%i $\to$ %i)" % (grads[key].shape[1], grads[key].shape[0]) if len(grads[key].shape) > 1 else ""
        )
        # key_ax.set_title(f"Layer {key} - {net.layers[key].__class__.__name__}")
        key_ax.set_title(f"{key} {hidden_dim_str}")
        fig_index += 1
    fig.suptitle(f"Gradient distribution for activation function {net.hparams.act_fn}", fontsize=14)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()
    plt.close()


def visualize_weights(
    net,
    fun_control,
    batch_size=128,
    device="cpu",
    color="C0",
    xlabel=None,
    stat="count",
    use_kde=True,
    columns=2,
    absolute=True,
    cmap="gray",
    figsize=(6, 6),
) -> None:
    """Visualize the activations of a neural network.

    Args:
        net (object):
            A neural network.
        fun_control (dict):
            A dictionary with the function control parameters.
        color (str, optional):
            The color to use. Defaults to "C0".

    Notes:
        Code is based on: PyTorch Lightning TUTORIAL 2: ACTIVATION FUNCTIONS, Author: Phillip Lippe,  License: CC BY-SA.

    """
    weights = {}
    for name, param in net.named_parameters():
        if name.endswith(".bias"):
            continue
        key_name = f"Layer {name.split('.')[1]}"
        weights[key_name] = param.detach().view(-1).cpu().numpy()
    print(f"weights: {weights}")

    # Plotting
    if cmap == "gray":
        cmap = "gray"
    elif cmap == "BlueWhiteRed":
        cmap = colors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
    else:
        cmap = "viridis"

    for layer, values in weights.items():
        n = int(math.sqrt(len(values)))
        if n * n != len(values):  # if the length is not a perfect square
            n += 1  # increase n by 1
            padding = np.zeros(n * n - len(values))  # create a zero array for padding
            values = np.concatenate((values, padding))  # append the padding to the values

        if absolute:
            reshaped_values = np.abs(values.reshape((n, n)))
        else:
            reshaped_values = values.reshape((n, n))

        plt.figure(figsize=figsize)
        plt.imshow(reshaped_values, cmap=cmap)  # use colormap to indicate the values
        plt.colorbar(label="Value")
        plt.title(f"Weights Plot for {layer}")
        plt.show()


def visualize_gradients(
    net,
    fun_control,
    batch_size=128,
    device="cpu",
    color="C0",
    xlabel=None,
    stat="count",
    use_kde=True,
    columns=2,
    absolute=True,
    cmap="gray",
    figsize=(6, 6),
) -> None:
    """Visualize the activations of a neural network.

    Args:
        net (object):
            A neural network.
        fun_control (dict):
            A dictionary with the function control parameters.
        color (str, optional):
            The color to use. Defaults to "C0".

    Notes:
        Code is based on: PyTorch Lightning TUTORIAL 2: ACTIVATION FUNCTIONS, Author: Phillip Lippe,  License: CC BY-SA.

    """
    grads = {}
    net.eval()
    dataset = fun_control["data_set"]
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # for batch in dataloader:
    #     inputs, targets = batch
    # small_loader = data.DataLoader(train_set, batch_size=1024)
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)
    # Pass one batch through the network, and calculate the gradients for the weights
    net.zero_grad()
    preds = net(inputs)
    loss = F.mse_loss(preds, targets)
    # loss = F.cross_entropy(preds, labels)  # Same as nn.CrossEntropyLoss, but as a function instead of module
    loss.backward()
    # We limit our visualization to the weight parameters and exclude the bias to reduce the number of plots
    grads = {
        name: params.grad.view(-1).cpu().clone().numpy() for name, params in net.named_parameters() if "weight" in name
    }
    net.zero_grad()
    print(f"grads: {grads}")

    # Plotting
    if cmap == "gray":
        cmap = "gray"
    elif cmap == "BlueWhiteRed":
        cmap = colors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
    else:
        cmap = "viridis"

    for layer, values in grads.items():
        n = int(math.sqrt(len(values)))
        if n * n != len(values):  # if the length is not a perfect square
            n += 1  # increase n by 1
            padding = np.zeros(n * n - len(values))  # create a zero array for padding
            values = np.concatenate((values, padding))  # append the padding to the values

        if absolute:
            reshaped_values = np.abs(values.reshape((n, n)))
        else:
            reshaped_values = values.reshape((n, n))

        plt.figure(figsize=figsize)
        plt.imshow(reshaped_values, cmap=cmap)  # use colormap to indicate the values
        plt.colorbar(label="Value")
        plt.title(f"Gradients Plot for {layer}")
        plt.show()
