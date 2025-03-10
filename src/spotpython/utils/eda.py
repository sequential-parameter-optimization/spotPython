from tabulate import tabulate
from spotpython.hyperparameters.values import (
    get_default_values,
    get_bound_values,
    get_var_name,
    get_var_type,
    get_transform,
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from spotpython.utils.time import get_timestamp


def get_stars(input_list) -> list:
    """Converts a list of values to a list of stars, which can be used to
        visualize the importance of a variable.

    Args:
        input_list (list): A list of values.

    Returns:
        (list):
            A list of strings.

    Examples:
        >>> from spotpython.utils.eda import convert_list
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


def print_exp_table(fun_control: dict, tablefmt="github", print_tab=True) -> str:
    """Generates a table with the design variables and their bounds.
        Can be used for the experiment design, which was not run yet.
    Args:
        fun_control (dict):
            A dictionary with function design variables.
        tablefmt (str):
            The format of the table. Defaults to "github".
        print_tab (bool):
            If True, the table is printed. Otherwise, the result code from tabulate
            is returned. Defaults to True.

    Returns:
        (str):
            a table with the design variables, their default values, and their bounds.
            Use the `print` function to display the table.

    Examples:
            >>> from spotpython.data.diabetes import Diabetes
                from spotpython.hyperdict.light_hyper_dict import LightHyperDict
                from spotpython.fun.hyperlight import HyperLight
                from spotpython.utils.init import fun_control_init
                from spotpython.spot import Spot
                from spotpython.utils.eda import print_exp_table
                fun_control = fun_control_init(
                    PREFIX="print_exp_table",
                    fun_evals=10,
                    max_time=1,
                    data_set = Diabetes(),
                    core_model_name="light.regression.NNLinearRegressor",
                    hyperdict=LightHyperDict,
                    _L_in=10,
                    _L_out=1)
                fun = HyperLight().fun
                print_exp_table(fun_control)
                | name           | type   | default   |   lower |   upper | transform             |
                |----------------|--------|-----------|---------|---------|-----------------------|
                | l1             | int    | 3         |     3   |    8    | transform_power_2_int |
                | epochs         | int    | 4         |     4   |    9    | transform_power_2_int |
                | batch_size     | int    | 4         |     1   |    4    | transform_power_2_int |
                | act_fn         | factor | ReLU      |     0   |    5    | None                  |
                | optimizer      | factor | SGD       |     0   |   11    | None                  |
                | dropout_prob   | float  | 0.01      |     0   |    0.25 | None                  |
                | lr_mult        | float  | 1.0       |     0.1 |   10    | None                  |
                | patience       | int    | 2         |     2   |    6    | transform_power_2_int |
                | batch_norm     | factor | 0         |     0   |    1    | None                  |
                | initialization | factor | Default   |     0   |    4    | None             |
    """
    default_values = get_default_values(fun_control)
    defaults = list(default_values.values())
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
    if print_tab:
        print(tab)
    else:
        return tab


def print_res_table(spot: object = None, tablefmt="github", print_tab=True) -> str:
    """
    Generates a table with the design variables and their bounds,
    after the run was completed.

    Args:
        spot (object):
            A spot object. Defaults to None.
        tablefmt (str):
            The format of the table. Defaults to "github".
        print_tab (bool):
            If True, the table is printed. Otherwise, the result code from tabulate
            is returned. Defaults to True.

    Returns:
        (str):
            a table with the design variables, their default values, their bounds,
            the value and the importance of each hyperparameter.
            Use the `print` function to display the table.

    Examples:
    >>> from spotpython.data.diabetes import Diabetes
        from spotpython.hyperdict.light_hyper_dict import LightHyperDict
        from spotpython.fun.hyperlight import HyperLight
        from spotpython.utils.init import fun_control_init, design_control_init
        from spotpython.spot import Spot
        from spotpython.utils.eda import print_res_table
        from spotpython.hyperparameters.values import set_hyperparameter
        fun_control = fun_control_init(
            PREFIX="print_res_table",
            fun_evals=5,
            max_time=1,
            data_set = Diabetes(),
            core_model_name="light.regression.NNLinearRegressor",
            hyperdict=LightHyperDict,
            _L_in=10,
            _L_out=1)
        set_hyperparameter(fun_control, "optimizer", [ "Adadelta", "Adam", "Adamax"])
        set_hyperparameter(fun_control, "l1", [1,2])
        set_hyperparameter(fun_control, "epochs", [2,2])
        set_hyperparameter(fun_control, "batch_size", [4,11])
        set_hyperparameter(fun_control, "dropout_prob", [0.0, 0.025])
        set_hyperparameter(fun_control, "patience", [1,2])
        design_control = design_control_init(init_size=3)
        fun = HyperLight().fun
        S = Spot(fun=fun, fun_control=fun_control, design_control=design_control)
        S.run()
        print_res_table(S)
        | name           | type   | default   |   lower |   upper | tuned                | transform             |   importance | stars   |
        |----------------|--------|-----------|---------|---------|----------------------|-----------------------|--------------|---------|
        | l1             | int    | 3         |     1.0 |     2.0 | 2.0                  | transform_power_2_int |        29.49 | *       |
        | epochs         | int    | 4         |     2.0 |     2.0 | 2.0                  | transform_power_2_int |         0.00 |         |
        | batch_size     | int    | 4         |     4.0 |    11.0 | 5.0                  | transform_power_2_int |         1.18 | *       |
        | act_fn         | factor | ReLU      |     0.0 |     5.0 | ELU                  | None                  |         0.32 | .       |
        | optimizer      | factor | SGD       |     0.0 |     2.0 | Adam                 | None                  |         0.08 |         |
        | dropout_prob   | float  | 0.01      |     0.0 |   0.025 | 0.010464684336704316 | None                  |         0.27 | .       |
        | lr_mult        | float  | 1.0       |     0.1 |    10.0 | 8.82569482726512     | None                  |         9.55 | *       |
        | patience       | int    | 2         |     1.0 |     2.0 | 2.0                  | transform_power_2_int |       100.00 | ***     |
        | batch_norm     | factor | 0         |     0.0 |     1.0 | 0                    | None                  |         0.05 |         |
        | initialization | factor | Default   |     0.0 |     4.0 | kaiming_normal       | None                  |         1.07 | *       |
    """
    fun_control = spot.fun_control
    default_values = get_default_values(fun_control)
    defaults = list(default_values.values())
    # try spot.print_results. If it fails, issue an error message that asked to run the spot object first
    try:
        res = spot.print_results(print_screen=False, dict=fun_control)
    except AttributeError as e:
        print(f"AttributeError: {e}. Did you run the spot object?")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return
    tuned = [item[1] for item in res]
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
    if print_tab:
        print(tab)
    else:
        return tab


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
        >>> from spotpython.utils.eda import compare_two_tree_models
        >>> from spotpython.hyperparameters.values import get_default_values
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
        >>> from spotpython.hyperparameters.values import get_one_config_from_X
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


def plot_missing_data(df, relative=False, figsize=(7, 5), color="grey", xlabel="Missing Data", title="Missing Data") -> None:
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
