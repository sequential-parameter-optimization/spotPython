from tabulate import tabulate
from spotPython.hyperparameters.values import (
    get_default_values,
    get_bound_values,
    get_var_name,
    get_var_type,
    get_transform,
)
import torch
from spotPython.light.csvdataset import CSVDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import seaborn as sns


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
        res = spot.print_results(print_screen=False)
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


def generate_config_id(config):
    """Generates a unique id for a configuration.
    Args:
        config (dict):
            A dictionary with the configuration.
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
        config_id += str(config[key]) + "_"
    return config_id[:-1]


def visualize_activations(net, device="cpu", color="C0"):
    """Visualizes the activations of a neural network.
    Code is based on: PyTorch Lightning TUTORIAL 2: ACTIVATION FUNCTIONS, Author: Phillip Lippe,
        License: CC BY-SA.

    Args:
        net (object):
            A neural network.
        device (str, optional):
            The device to use. Defaults to "cpu".
        color (str, optional):
            The color to use. Defaults to "C0".
    Examples:
        >>> from spotPython.hyperparameters.values import get_one_config_from_X
        >>> X = spot_tuner.to_all_dim(spot_tuner.min_X.reshape(1,-1))
        >>> config = get_one_config_from_X(X, fun_control)
        >>> model = fun_control["core_model"](**config, _L_in=64, _L_out=11)
        >>> visualize_activations(model, device="cpu", color=f"C{0}")
    """
    activations = {}
    net.eval()
    # Create an instance of CSVDataset
    dataset = CSVDataset(csv_file="./data/VBDP/train.csv", train=True)
    # Set batch size for DataLoader
    batch_size = 128
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # for batch in dataloader:
    #     inputs, targets = batch
    # small_loader = data.DataLoader(train_set, batch_size=1024)
    inputs, _ = next(iter(dataloader))
    with torch.no_grad():
        layer_index = 0
        inputs = inputs.to(device)
        inputs = inputs.view(inputs.size(0), -1)
        # We need to manually loop through the layers to save all activations
        for layer_index, layer in enumerate(net.layers[:-1]):
            inputs = layer(inputs)
            activations[layer_index] = inputs.view(-1).cpu().numpy()

    # Plotting
    columns = 4
    rows = math.ceil(len(activations) / columns)
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
