import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.colors as colors
from spotpython.hyperparameters.values import get_tuned_architecture
from spotpython.light.trainmodel import train_model
from spotpython.light.loadmodel import load_light_from_checkpoint
from spotpython.utils.classes import get_removed_attributes_and_base_net
import pandas as pd
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
from matplotlib.ticker import MaxNLocator


def get_activations(net, fun_control, batch_size, device="cpu") -> dict:
    """
    Get the activations of a neural network.

    Args:
        net (object):
            A neural network.
        fun_control (dict):
            A dictionary with the function control.
        batch_size (int, optional):
            The batch size.
        device (str, optional):
            The device to use. Defaults to "cpu".

    Returns:
        dict: A dictionary with the activations of the neural network.

    Examples:
        >>> from torch.utils.data import DataLoader
            from spotpython.utils.init import fun_control_init
            from spotpython.hyperparameters.values import set_control_key_value
            from spotpython.data.diabetes import Diabetes
            from spotpython.light.regression.netlightregression import NetLightRegression
            from spotpython.hyperdict.light_hyper_dict import LightHyperDict
            from spotpython.hyperparameters.values import add_core_model_to_fun_control
            from spotpython.hyperparameters.values import (
                    get_default_hyperparameters_as_array, get_one_config_from_X)
            from spotpython.hyperparameters.values import set_control_key_value
            from spotpython.plot.xai import get_activations
            fun_control = fun_control_init(
                _L_in=10, # 10: diabetes
                _L_out=1,
                )
            dataset = Diabetes()
            set_control_key_value(control_dict=fun_control,
                                    key="data_set",
                                    value=dataset,
                                    replace=True)
            add_core_model_to_fun_control(fun_control=fun_control,
                                        core_model=NetLightRegression,
                                        hyper_dict=LightHyperDict)
            X = get_default_hyperparameters_as_array(fun_control)
            config = get_one_config_from_X(X, fun_control)
            _L_in = fun_control["_L_in"]
            _L_out = fun_control["_L_out"]
            model = fun_control["core_model"](**config, _L_in=_L_in, _L_out=_L_out)
            batch_size= config["batch_size"]
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            get_activations(model, fun_control=fun_control, batch_size=batch_size, device = "cpu")
            {0: array([ 1.43207282e-01,  6.29711570e-03,  1.04200505e-01, -3.79187055e-03,
                        -1.74976081e-01, -7.97475874e-02, -2.00860098e-01,  2.48444706e-01, ...

    """
    activations = {}
    net.eval()
    print(f"net: {net}")
    dataset = fun_control["data_set"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    inputs, _ = next(iter(dataloader))
    with torch.no_grad():
        layer_index = 0
        inputs = inputs.to(device)
        inputs = inputs.view(inputs.size(0), -1)
        for layer_index, layer in enumerate(net.layers):
            inputs = layer(inputs)
            if isinstance(layer, nn.Linear):
                activations[layer_index] = inputs.view(-1).cpu().numpy()
    # print(f"activations:{activations}")
    return activations


def get_weights(net, return_index=False) -> dict:
    """
    Get the weights of a neural network.

    Args:
        net (object):
            A neural network.
        return_index (bool, optional):
            Whether to return the index. Defaults to False.

    Returns:
        dict:
            A dictionary with the weights of the neural network.
        index (list):
            The layer index list.

    Examples:
        >>> from torch.utils.data import DataLoader
            from spotpython.utils.init import fun_control_init
            from spotpython.hyperparameters.values import set_control_key_value
            from spotpython.data.diabetes import Diabetes
            from spotpython.light.regression.netlightregression import NetLightRegression
            from spotpython.hyperdict.light_hyper_dict import LightHyperDict
            from spotpython.hyperparameters.values import add_core_model_to_fun_control
            from spotpython.hyperparameters.values import (
                    get_default_hyperparameters_as_array, get_one_config_from_X)
            from spotpython.hyperparameters.values import set_control_key_value
            from spotpython.plot.xai import get_activations
            fun_control = fun_control_init(
                _L_in=10, # 10: diabetes
                _L_out=1,
                )
            dataset = Diabetes()
            set_control_key_value(control_dict=fun_control,
                                    key="data_set",
                                    value=dataset,
                                    replace=True)
            add_core_model_to_fun_control(fun_control=fun_control,
                                        core_model=NetLightRegression,
                                        hyper_dict=LightHyperDict)
            X = get_default_hyperparameters_as_array(fun_control)
            config = get_one_config_from_X(X, fun_control)
            _L_in = fun_control["_L_in"]
            _L_out = fun_control["_L_out"]
            model = fun_control["core_model"](**config, _L_in=_L_in, _L_out=_L_out)
            batch_size= config["batch_size"]
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            get_weights(model)
            {'Layer 0': array([-0.12895013,  0.01047492, -0.15705723,  0.11925378, -0.26944348,
                                0.23180881, -0.22984707, -0.25141433, -0.19982024,  0.1432175 ,
                                -0.11684369,  0.11833665, -0.2683918 , -0.19186287, -0.11611126,
                                -0.06214499, -0.2412386 ,  0.20706299, -0.07457635,  0.10150522,
                                0.22361842,  0.05891514,  0.08647272,  0.3052416 , -0.1426217 ,
                                0.10016555, -0.14069483,  0.22599205,  0.25255737, -0.29155323,
                                0.2699465 ,  0.1510033 ,  0.13780165,  0.13018301,  0.26287982,
                                -0.04175457, -0.26743335, -0.09074122, -0.2227112 ,  0.02090478,
                                -0.0590421 , -0.16961981, -0.02875188,  0.2995954 , -0.02494261,
                                0.01004025, -0.04931906,  0.04971322,  0.28176293,  0.19337103,
                                0.11224869,  0.06871963,  0.07456425,  0.12216929, -0.04086405,
                                -0.29390487, -0.19555901,  0.26992753,  0.01890203, -0.25616774,
                                0.04987782,  0.26129004, -0.29883513, -0.21289697, -0.12594265,
                                0.0126926 , -0.07375361, -0.03475064, -0.30828732,  0.14808285,
                                0.27756676,  0.19329056, -0.22393112, -0.25491226,  0.13131431,
                                0.00710201,  0.12963155, -0.3090024 , -0.01885444,  0.22301766],
                            dtype=float32),

    """
    weights = {}
    index = []
    for name, param in net.named_parameters():
        if name.endswith(".bias"):
            continue
        # add (int(name.split(".")[1])) to the index list
        index.append(int(name.split(".")[1]))
        key_name = f"Layer {name.split('.')[1]}"
        weights[key_name] = param.detach().view(-1).cpu().numpy()
    # print(f"weights: {weights}")
    if return_index:
        return weights, index
    else:
        return weights


def get_gradients(net, fun_control, batch_size, device="cpu") -> dict:
    """
    Get the gradients of a neural network.

    Args:
        net (object):
            A neural network.
        fun_control (dict):
            A dictionary with the function control.
        batch_size (int, optional):
            The batch size.
        device (str, optional):
            The device to use. Defaults to "cpu".

    Returns:
        dict: A dictionary with the gradients of the neural network.

    Examples:
        >>> from torch.utils.data import DataLoader
            from spotpython.utils.init import fun_control_init
            from spotpython.hyperparameters.values import set_control_key_value
            from spotpython.data.diabetes import Diabetes
            from spotpython.light.regression.netlightregression import NetLightRegression
            from spotpython.hyperdict.light_hyper_dict import LightHyperDict
            from spotpython.hyperparameters.values import add_core_model_to_fun_control
            from spotpython.hyperparameters.values import (
                    get_default_hyperparameters_as_array, get_one_config_from_X)
            from spotpython.hyperparameters.values import set_control_key_value
            from spotpython.plot.xai import get_activations
            fun_control = fun_control_init(
                _L_in=10, # 10: diabetes
                _L_out=1,
                )
            dataset = Diabetes()
            set_control_key_value(control_dict=fun_control,
                                    key="data_set",
                                    value=dataset,
                                    replace=True)
            add_core_model_to_fun_control(fun_control=fun_control,
                                        core_model=NetLightRegression,
                                        hyper_dict=LightHyperDict)
            X = get_default_hyperparameters_as_array(fun_control)
            config = get_one_config_from_X(X, fun_control)
            _L_in = fun_control["_L_in"]
            _L_out = fun_control["_L_out"]
            model = fun_control["core_model"](**config, _L_in=_L_in, _L_out=_L_out)
            batch_size= config["batch_size"]
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            get_gradients(model, fun_control=fun_control, batch_size=batch_size, device = "cpu")
            {'layers.0.weight': array([ 0.10417588, -0.04161512,  0.10597267,  0.02180895,  0.12001498,
                    0.02890352,  0.0114617 ,  0.08183316,  0.2495192 ,  0.5108763 ,
                    0.14668094, -0.07902834,  0.00912531,  0.02640062,  0.14108546, ...
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
    # TODO: Add more loss functions
    loss = F.mse_loss(preds, targets)
    # loss = F.cross_entropy(preds, labels)  # Same as nn.CrossEntropyLoss, but as a function instead of module
    loss.backward()
    # We limit our visualization to the weight parameters and exclude the bias to reduce the number of plots
    grads = {
        name: params.grad.view(-1).cpu().clone().numpy() for name, params in net.named_parameters() if "weight" in name
    }
    net.zero_grad()
    return grads


def plot_nn_values_hist(nn_values, net, nn_values_names="", color="C0", columns=2) -> None:
    """
    Plot the values of a neural network.
    Can be used to plot the weights, gradients, or activations of a neural network.

    Args:
        nn_values (dict):
            A dictionary with the values of the neural network. For example,
            the weights, gradients, or activations.
        net (object):
            A neural network.
        color (str, optional):
            The color to use. Defaults to "C0".
        columns (int, optional):
            The number of columns. Defaults to 2.

    """
    n = len(nn_values)
    print(f"n:{n}")
    rows = n // columns + int(n % columns > 0)
    fig, ax = plt.subplots(rows, columns, figsize=(columns * 2.7, rows * 2.5))
    fig_index = 0
    for key in nn_values:
        key_ax = ax[fig_index // columns][fig_index % columns]
        sns.histplot(data=nn_values[key], bins=50, ax=key_ax, color=color, kde=True, stat="density")
        hidden_dim_str = (
            r"(%i $\to$ %i)" % (nn_values[key].shape[1], nn_values[key].shape[0])
            if len(nn_values[key].shape) > 1
            else ""
        )
        key_ax.set_title(f"{key} {hidden_dim_str}")
        # key_ax.set_title(f"Layer {key} - {net.layers[key].__class__.__name__}")
        fig_index += 1
    fig.suptitle(f"{nn_values_names} distribution for activation function {net.hparams.act_fn}", fontsize=14)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()


def old_plot_nn_values_scatter(
    nn_values, nn_values_names="", absolute=True, cmap="gray", figsize=(6, 6), return_reshaped=False
):
    """
    Plot the values of a neural network.
    Can be used to plot the weights, gradients, or activations of a neural network.

    Args:
        nn_values (dict):
            A dictionary with the values of the neural network. For example,
            the weights, gradients, or activations.
        nn_values_names (str, optional):
            The name of the values. Defaults to "".
        absolute (bool, optional):
            Whether to use the absolute values. Defaults to True.
        cmap (str, optional):
            The colormap to use. Defaults to "gray".
        figsize (tuple, optional):
            The figure size. Defaults to (6, 6).
        return_reshaped (bool, optional):
            Whether to return the reshaped values. Defaults to False.

    """
    if cmap == "gray":
        cmap = "gray"
    elif cmap == "BlueWhiteRed":
        cmap = colors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
    else:
        cmap = "viridis"

    res = {}
    for layer, values in nn_values.items():
        k = len(values)
        print(f"{k} values in Layer {layer}.")
        if is_square(k):
            n = int(math.sqrt(k))
        else:
            n = int(math.sqrt(len(values)) + 1)
            padding = np.zeros(n * n - len(values))  # create a zero array for padding
            print(f"{len(padding)} padding values added.")
            values = np.concatenate((values, padding))  # append the padding to the values

        print(f"{len(values)} values in Layer {layer}.")
        if absolute:
            reshaped_values = np.abs(values.reshape((n, n)))
        else:
            reshaped_values = values.reshape((n, n))

        plt.figure(figsize=figsize)
        plt.imshow(reshaped_values, cmap=cmap)  # use colormap to indicate the values
        plt.colorbar(label="Value")
        plt.title(f"{nn_values_names} Plot for {layer}")
        plt.show()
        # add reshaped_values to the dictionary res
        res[layer] = reshaped_values
    if return_reshaped:
        return res


def plot_nn_values_scatter(
    nn_values, nn_values_names="", absolute=True, cmap="gray", figsize=(6, 6), return_reshaped=False, show=True
) -> dict:
    """
    Plot the values of a neural network including a marker for padding values.
    For simplicity, this example will annotate 'P' directly on the plot for padding values
    using a unique marker value approach.

    Args:
        nn_values (dict):
            A dictionary with the values of the neural network. For example,
            the weights, gradients, or activations.
        nn_values_names (str, optional):
            The name of the values. Defaults to "".
        absolute (bool, optional):
            Whether to use the absolute values. Defaults to True.
        cmap (str, optional):
            The colormap to use. Defaults to "gray".
        figsize (tuple, optional):
            The figure size. Defaults to (6, 6).
        return_reshaped (bool, optional):
            Whether to return the reshaped values. Defaults to False.
        show (bool, optional):
            Whether to show the plot. Defaults to True.

    Returns:
        dict: A dictionary with the reshaped values.
    """
    if cmap == "gray":
        cmap = "gray"
    elif cmap == "BlueWhiteRed":
        cmap = colors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
    else:
        cmap = "viridis"

    res = {}
    padding_marker = np.nan  # Use NaN as a special marker for padding
    for layer, values in nn_values.items():
        k = len(values)
        print(f"{k} values in Layer {layer}.")
        n = int(math.sqrt(k))
        if n * n != k:  # if the length is not a perfect square
            n += 1  # Adjust n for padding
            print(f"{n*n-k} padding values added.")
            values = np.append(values, [padding_marker] * (n * n - k))  # Append padding values

        print(f"{len(values)} values now in Layer {layer}.")

        if absolute:
            reshaped_values = np.abs(values).reshape((n, n))
            # Mark padding values distinctly by setting them back to NaN
            reshaped_values[reshaped_values == np.abs(padding_marker)] = np.nan
        else:
            reshaped_values = values.reshape((n, n))

        _, ax = plt.figure(figsize=figsize), plt.gca()
        cax = ax.imshow(reshaped_values, cmap=cmap, interpolation="nearest")
        for i in range(n):
            for j in range(n):
                if np.isnan(reshaped_values[i, j]):
                    ax.text(j, i, "P", ha="center", va="center", color="red")
        plt.colorbar(cax, label="Value")
        plt.title(f"{nn_values_names} Plot for {layer}")
        if show:
            plt.show()
        # Add reshaped_values to the dictionary res
        res[layer] = reshaped_values
    if return_reshaped:
        return res


def visualize_activations_distributions(net, fun_control, batch_size, device="cpu", color="C0", columns=2) -> None:
    """
    Plots a histogram of the activations of a neural network.

    Args:
        net (object):
            A neural network.
        fun_control (dict):
            A dictionary with the function control.
        batch_size (int, optional):
            The batch size.
        device (str, optional):
            The device to use. Defaults to "cpu".
        color (str, optional):
            The color to use. Defaults to "C0".
        columns (int, optional):
            The number of columns. Defaults to 2.

    Returns:
        None

    """
    activations = get_activations(net, fun_control, batch_size, device)
    plot_nn_values_hist(activations, net, nn_values_names="Activations", color=color, columns=columns)


def visualize_weights_distributions(net, color="C0", columns=2) -> None:
    """
    Plot the weights distributions of a neural network.

    Args:
        net (object):
            A neural network.
        color (str, optional):
            The color to use. Defaults to "C0".
        columns (int, optional):
            The number of columns. Defaults to 2.

    Returns:
        None

    """
    weights = get_weights(net)
    plot_nn_values_hist(weights, net, nn_values_names="Weights", color=color, columns=columns)


def visualize_gradient_distributions(
    net, fun_control, batch_size, device="cpu", color="C0", xlabel=None, stat="count", use_kde=True, columns=2
) -> None:
    """
    Plot the gradients distributions of a neural network.

    Args:
        net (object):
            A neural network.
        fun_control (dict):
            A dictionary with the function control.
        batch_size (int, optional):
            The batch size.
        device (str, optional):
            The device to use. Defaults to "cpu".
        color (str, optional):
            The color to use. Defaults to "C0".
        xlabel (str, optional):
            The x label. Defaults to None.
        stat (str, optional):
            The stat. Defaults to "count".
        use_kde (bool, optional):
            Whether to use kde. Defaults to True.
        columns (int, optional):
            The number of columns. Defaults to 2.

    Returns:
        None

    """
    grads = get_gradients(net, fun_control, batch_size, device)
    plot_nn_values_hist(grads, net, nn_values_names="Gradients", color=color, columns=columns)


def visualize_activations(net, fun_control, batch_size, device, absolute=True, cmap="gray", figsize=(6, 6)) -> None:
    """
    Scatter plots the activations of a neural network.

    Args:
        net (object):
            A neural network.
        fun_control (dict):
            A dictionary with the function control.
        batch_size (int, optional):
            The batch size.
        device (str, optional):
            The device to use.
        absolute (bool, optional):
            Whether to use the absolute values. Defaults to True.
        cmap (str, optional):
            The colormap to use. Defaults to "gray".
        figsize (tuple, optional):
            The figure size. Defaults to (6, 6).

    Returns:
        None

    """
    activations = get_activations(net, fun_control, batch_size, device)
    plot_nn_values_scatter(
        nn_values=activations, nn_values_names="Activations", absolute=absolute, cmap=cmap, figsize=figsize
    )


def visualize_weights(net, absolute=True, cmap="gray", figsize=(6, 6)) -> None:
    """
    Scatter plots the weights of a neural network.

    Args:
        net (object):
            A neural network.
        absolute (bool, optional):
            Whether to use the absolute values. Defaults to True.
        cmap (str, optional):
            The colormap to use. Defaults to "gray".
        figsize (tuple, optional):
            The figure size. Defaults to (6, 6).

    Returns:
        None

    """
    weights = get_weights(net)
    plot_nn_values_scatter(nn_values=weights, nn_values_names="Weights", absolute=absolute, cmap=cmap, figsize=figsize)


def visualize_gradients(net, fun_control, batch_size, absolute=True, cmap="gray", figsize=(6, 6)) -> None:
    """
    Scatter plots the gradients of a neural network.

    Args:
        net (object):
            A neural network.
        fun_control (dict):
            A dictionary with the function control.
        batch_size (int, optional):
            The batch size.
        absolute (bool, optional):
            Whether to use the absolute values. Defaults to True.
        cmap (str, optional):
            The colormap to use. Defaults to "gray".
        figsize (tuple, optional):
            The figure size. Defaults to (6, 6).

    Returns:
        None
    """
    grads = get_gradients(
        net,
        fun_control,
        batch_size=batch_size,
    )
    plot_nn_values_scatter(nn_values=grads, nn_values_names="Gradients", absolute=absolute, cmap=cmap, figsize=figsize)


def get_attributions(
    spot_tuner,
    fun_control,
    attr_method="IntegratedGradients",
    baseline=None,
    abs_attr=True,
    n_rel=5,
) -> pd.DataFrame:
    """Get the attributions of a neural network.

    Args:
        spot_tuner (object):
            The spot tuner object.
        fun_control (dict):
            A dictionary with the function control.
        attr_method (str, optional):
            The attribution method. Defaults to "IntegratedGradients".
        baseline (torch.Tensor, optional):
            The baseline for the attribution methods. Defaults to None.
        abs_attr (bool, optional):
            Whether the method should sort by the absolute attribution values. Defaults to True.
        n_rel (int, optional):
            The number of relevant features. Defaults to 5.

    Returns:
        pd.DataFrame (object): A DataFrame with the attributions.
    """
    try:
        fun_control["data_set"].names
    except AttributeError:
        fun_control["data_set"].names = None
    feature_names = fun_control["data_set"].names
    total_attributions = None
    config = get_tuned_architecture(spot_tuner, fun_control)
    train_model(config, fun_control, timestamp=False)
    model_loaded = load_light_from_checkpoint(config, fun_control, postfix="_TRAIN")
    removed_attributes, model = get_removed_attributes_and_base_net(net=model_loaded)
    model = model.to("cpu")
    model.eval()
    dataset = fun_control["data_set"]
    try:
        n_features = dataset.data.shape[1]
    except AttributeError:
        n_features = dataset.tensors[0].shape[1]
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]
    batch_size = config["batch_size"]
    # train_loader = DataLoader(dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset, batch_size=batch_size)
    if attr_method == "IntegratedGradients":
        attr = IntegratedGradients(model)
    elif attr_method == "DeepLift":
        attr = DeepLift(model)
    elif attr_method == "GradientShap":  # Todo: would need a baseline
        if baseline is None:
            raise ValueError("baseline cannot be 'None' for GradientShap")
        attr = GradientShap(model)
    elif attr_method == "FeatureAblation":
        attr = FeatureAblation(model)
    else:
        raise ValueError(
            """
            Unsupported attribution method.
            Please choose from 'IntegratedGradients', 'DeepLift', 'GradientShap', or 'FeatureAblation'.
            """
        )
    for inputs, labels in test_loader:
        attributions = attr.attribute(inputs, return_convergence_delta=False, baselines=baseline)
        if total_attributions is None:
            total_attributions = attributions
        else:
            if len(attributions) == len(total_attributions):
                total_attributions += attributions

    # Calculation of average attribution across all batches
    avg_attributions = total_attributions.mean(dim=0).detach().numpy()

    # Transformation to the absolute attribution values if abs_attr is True
    # Get indices of the n most important features
    if abs_attr is True:
        abs_avg_attributions = abs(avg_attributions)
        top_n_indices = abs_avg_attributions.argsort()[-n_rel:][::-1]
    else:
        top_n_indices = avg_attributions.argsort()[-n_rel:][::-1]

    # Get the importance values for the top n features
    top_n_importances = avg_attributions[top_n_indices]

    df = pd.DataFrame(
        {
            "Feature Index": top_n_indices,
            "Feature": [feature_names[i] for i in top_n_indices],
            attr_method + "Attribution": top_n_importances,
        }
    )
    return df


def plot_attributions(df, attr_method="IntegratedGradients") -> None:
    """
    Plot the attributions of a neural network.

    Args:
        df (pd.DataFrame):
            A DataFrame with the attributions.
        attr_method (str, optional):
            The attribution method. Defaults to "IntegratedGradients".

    Returns:
        None

    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=attr_method + "Attribution", y="Feature", data=df, palette="viridis", hue="Feature")
    plt.title(f"Top {df.shape[0]} Features by {attr_method} Attribution")
    plt.xlabel(f"{attr_method} Attribution Value")
    plt.ylabel("Feature")
    plt.show()


def is_square(n) -> bool:
    """Check if a number is a square number.

    Args:
        n (int): The number to check.

    Returns:
        bool: True if the number is a square number, False otherwise.

    Examples:
        >>> is_square(4)
        True
        >>> is_square(5)
        False
    """
    return n == int(math.sqrt(n)) ** 2


def get_layer_conductance(spot_tuner, fun_control, layer_idx) -> np.ndarray:
    """
    Compute the average layer conductance attributions for a specified layer in the model.

    Args:
        spot_tuner (spot.Spot):
            The spot tuner object containing the trained model.
        fun_control (dict):
            The fun_control dictionary containing the hyperparameters used to train the model.
        layer_idx (int):
            Index of the layer for which to compute layer conductance attributions.

    Returns:
        numpy.ndarray:
            An array containing the average layer conductance attributions for the specified layer.
            The shape of the array corresponds to the shape of the attributions.
    """
    try:
        fun_control["data_set"].names
    except AttributeError:
        fun_control["data_set"].names = None
    feature_names = fun_control["data_set"].names

    config = get_tuned_architecture(spot_tuner, fun_control)
    train_model(config, fun_control, timestamp=False)
    model_loaded = load_light_from_checkpoint(config, fun_control, postfix="_TRAIN")
    removed_attributes, model = get_removed_attributes_and_base_net(net=model_loaded)
    model = model.to("cpu")
    model.eval()

    dataset = fun_control["data_set"]
    try:
        n_features = dataset.data.shape[1]
    except AttributeError:
        n_features = dataset.tensors[0].shape[1]
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]
    batch_size = config["batch_size"]
    # train_loader = DataLoader(dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset, batch_size=batch_size)

    total_layer_attributions = None
    layers = model.layers
    print("Conductance analysis for layer: ", layers[layer_idx])
    lc = LayerConductance(model, layers[layer_idx])

    for inputs, labels in test_loader:
        lc_attr_test = lc.attribute(inputs, n_steps=10, attribute_to_layer_input=True)
        if total_layer_attributions is None:
            total_layer_attributions = lc_attr_test
        else:
            if len(lc_attr_test) == len(total_layer_attributions):
                total_layer_attributions += lc_attr_test

    avg_layer_attributions = total_layer_attributions.mean(dim=0).detach().numpy()

    return avg_layer_attributions


def get_weights_conductance_last_layer(spot_tuner, fun_control):
    """
    Get the weights and the conductance of the last layer.
    """
    config = get_tuned_architecture(spot_tuner, fun_control)
    train_model(config, fun_control, timestamp=False)
    model_loaded = load_light_from_checkpoint(config, fun_control, postfix="_TRAIN")
    removed_attributes, model = get_removed_attributes_and_base_net(net=model_loaded)
    model = model.to("cpu")
    model.eval()

    weights, index = get_weights(model, return_index=True)
    layer_idx = index[-1]
    weights_last = weights[f"Layer {layer_idx}"]
    weights_last
    layer_conductance_last = get_layer_conductance(spot_tuner, fun_control, layer_idx=layer_idx)

    return weights_last, layer_conductance_last


def plot_conductance_last_layer(weights_last, layer_conductance_last, show=True):
    """
    Plot the conductance of the last layer.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(weights_last)), weights_last / weights_last.max(), label="Weights", alpha=0.5)
    ax.bar(
        range(len(layer_conductance_last)),
        layer_conductance_last / layer_conductance_last.max(),
        label="Layer Conductance",
        alpha=0.5,
    )
    ax.set_xlabel("Weight Index")
    ax.set_ylabel("Normalized Value")
    ax.set_title("Layer Conductance vs. Weights")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if show:
        plt.show()


def get_all_layers_conductance(spot_tuner, fun_control):
    config = get_tuned_architecture(spot_tuner, fun_control)
    train_model(config, fun_control, timestamp=False)
    model_loaded = load_light_from_checkpoint(config, fun_control, postfix="_TRAIN")
    removed_attributes, model = get_removed_attributes_and_base_net(net=model_loaded)
    model = model.to("cpu")
    model.eval()
    _, index = get_weights(model, return_index=True)
    layer_conductance = {}
    for i in index:
        layer_conductance[i] = get_layer_conductance(spot_tuner, fun_control, layer_idx=i)
    return layer_conductance
