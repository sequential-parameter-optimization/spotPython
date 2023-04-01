from tabulate import tabulate
from spotPython.hyperparameters.values import get_default_values, get_bound_values, get_var_name, get_var_type


def get_stars(input_list) -> list:
    """Converts a list of values to a list of stars, which can be used to
        visualize the importance of a variable.
    Args:
        input_list (list): A list of values.
    Returns:
        list: A list of strings.
    Example:
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
        fun_control (dict): A dictionary with function design variables.
        spot (object): A spot object. Defaults to None.
    Returns:
        str: A table with the design variables, their default values, and their bounds.
            If a spot object is provided,
            the table will also include the value
            and the importance of each hyperparameter.
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
            },
            headers="keys",
            tablefmt=tablefmt,
        )
    else:
        res = spot.print_results(print_screen=False)
        tuned = [item[1] for item in res]
        imp = spot.print_importance(threshold=0.0, print_screen=False)
        importance = [item[1] for item in imp]
        stars = get_stars(importance)
        tab = tabulate(
            {
                "name": get_var_name(fun_control),
                "type": get_var_type(fun_control),
                "default": defaults,
                "lower": get_bound_values(fun_control, "lower", as_list=True),
                "upper": get_bound_values(fun_control, "upper", as_list=True),
                "tuned": tuned,
                "importance": importance,
                "stars": stars,
            },
            headers="keys",
            numalign="right",
            floatfmt=("", "", "", "", "", "", ".2f"),
            tablefmt=tablefmt,
        )
    return tab
