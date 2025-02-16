import pandas as pd
import numpy as np
from scipy.stats import norm, t
from numpy.linalg import pinv, inv, LinAlgError
import copy
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols


def cov_to_cor(covariance_matrix) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix.

    Args:
        covariance_matrix (numpy.ndarray): A square matrix of covariance values.

    Returns:
        numpy.ndarray: A corresponding square matrix of correlation coefficients.

    Examples:
        >>> from spotpython.utils.stats import cov_to_cor
        >>> import numpy as np
        >>> cov_matrix = np.array([[1, 0.8], [0.8, 1]])
        >>> cov_to_cor(cov_matrix)
        array([[1. , 0.8],
               [0.8, 1. ]])
    """
    d = np.sqrt(np.diag(covariance_matrix))
    return covariance_matrix / np.outer(d, d)


def partial_correlation(x, method="pearson") -> dict:
    """Calculate the partial correlation matrix for a given data set.

    Args:
        x (pandas.DataFrame or numpy.ndarray): The data matrix with variables as columns.
        method (str): Correlation method, one of 'pearson', 'kendall', or 'spearman'.

    Returns:
        dict: A dictionary containing the partial correlation estimates, p-values,
            statistics, sample size (n), number of given parameters (gp), and method used.

    Raises:
        ValueError: If input is not a matrix-like structure or not numeric.

    References:
        1. Kim, S. ppcor: An R package for a fast calculation to semi-partial correlation coefficients.
        Commun Stat Appl Methods 22, 6 (Nov 2015), 665–674.

    Examples:
        >>> from spotpython.utils.stats import partial_correlation
        >>> import numpy as np
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        >>>     'A': [1, 2, 3],
        >>>     'B': [4, 5, 6],
        >>>     'C': [7, 8, 9]
        >>> })
        >>> partial_correlation(data, method='pearson')
        {'estimate': array([[ 1. , -1. ,  1. ],
                            [-1. ,  1. , -1. ],
                            [ 1. , -1. ,  1. ]]),
        'p_value': array([[0. , 0. , 0. ],
                          [0. , 0. , 0. ],
                          [0. , 0. , 0. ]]), ...
        }
    """
    eps = 1e-6
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    if not isinstance(x, np.ndarray):
        raise ValueError("Supply a matrix-like 'x'")
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("'x' must be numeric")

    n = x.shape[0]
    gp = x.shape[1] - 2
    cvx = np.cov(x, rowvar=False, bias=True)

    try:
        if np.linalg.det(cvx) < np.finfo(float).eps:
            icvx = pinv(cvx)
        else:
            icvx = inv(cvx)
    except LinAlgError:
        icvx = pinv(cvx)

    p_cor = -cov_to_cor(icvx)
    np.fill_diagonal(p_cor, 1)

    if method == "kendall":
        denominator = np.sqrt(2 * (2 * (n - gp) + 5) / (9 * (n - gp) * (n - 1 - gp)))
        statistic = p_cor / denominator
        p_value = 2 * norm.cdf(-np.abs(statistic))
    else:
        factor = np.sqrt((n - 2 - gp) / (1 + eps - p_cor**2))
        statistic = p_cor * factor
        p_value = 2 * t.cdf(-np.abs(statistic), df=n - 2 - gp)

    np.fill_diagonal(statistic, 0)
    np.fill_diagonal(p_value, 0)

    return {"estimate": p_cor, "p_value": p_value, "statistic": statistic, "n": n, "gp": gp, "method": method}


def partial_correlation_test(x, y, z, method="pearson") -> dict:
    """The partial correlation coefficient between x and y given z.
        x and y should be arrays (vectors) of the same length, and z should be a data frame (matrix).

    Args:
        x (array-like): The first variable as a 1-dimensional array or list.
        y (array-like): The second variable as a 1-dimensional array or list.
        z (pandas.DataFrame): A data frame containing other conditional variables.
        method (str): Correlation method, one of 'pearson', 'kendall', or 'spearman'.

    Returns:
        dict: A dictionary with the partial correlation estimate, p-value, statistic,
            sample size (n), number of given parameters (gp), and method used.

    References:
        1. Kim, S. ppcor: An R package for a fast calculation to semi-partial correlation coefficients.
        Commun Stat Appl Methods 22, 6 (Nov 2015), 665–674.

    Examples:
        >>> from spotpython.utils.stats import pairwise_partial_correlation
        >>> import pandas as pd
        >>> x = [1, 2, 3]
        >>> y = [4, 5, 6]
        >>> z = pd.DataFrame({'C': [7, 8, 9]})
        >>> pairwise_partial_correlation(x, y, z)
        {'estimate': -1.0, 'p_value': 0.0, 'statistic': -inf, 'n': 3, 'gp': 1, 'method': 'pearson'}
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = pd.DataFrame(z)

    xyz = pd.concat([pd.Series(x), pd.Series(y), z], axis=1)

    pcor_result = partial_correlation(xyz, method=method)

    return {
        "estimate": pcor_result["estimate"][0, 1],
        "p_value": pcor_result["p_value"][0, 1],
        "statistic": pcor_result["statistic"][0, 1],
        "n": pcor_result["n"],
        "gp": pcor_result["gp"],
        "method": method,
    }


def semi_partial_correlation(x, method="pearson"):
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    if not isinstance(x, np.ndarray):
        raise ValueError("Supply a matrix-like 'x'")
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("'x' must be numeric")

    n = x.shape[0]
    gp = x.shape[1] - 2
    cvx = np.cov(x, rowvar=False, bias=True)

    try:
        if np.linalg.det(cvx) < np.finfo(float).eps:
            icvx = pinv(cvx)
        else:
            icvx = inv(cvx)
    except LinAlgError:
        icvx = pinv(cvx)

    sp_cor = -cov_to_cor(icvx) / np.sqrt(np.diag(cvx)) / np.sqrt(np.abs(np.diag(icvx) - np.square(icvx.T) / np.diag(icvx)))
    np.fill_diagonal(sp_cor, 1)

    if method == "kendall":
        denominator = np.sqrt(2 * (2 * (n - gp) + 5) / (9 * (n - gp) * (n - 1 - gp)))
        statistic = sp_cor / denominator
        p_value = 2 * norm.cdf(-np.abs(statistic))
    else:
        factor = np.sqrt((n - 2 - gp) / (1 - sp_cor**2))
        statistic = sp_cor * factor
        p_value = 2 * t.cdf(-np.abs(statistic), df=n - 2 - gp)

    np.fill_diagonal(statistic, 0)
    np.fill_diagonal(p_value, 0)

    return {"estimate": sp_cor, "p_value": p_value, "statistic": statistic, "n": n, "gp": gp, "method": method}


def pairwise_semi_partial_correlation(x, y, z, method="pearson"):
    x = np.asarray(x)
    y = np.asarray(y)
    z = pd.DataFrame(z)

    xyz = pd.concat([pd.Series(x), pd.Series(y), z], axis=1)

    spcor_result = semi_partial_correlation(xyz, method=method)

    return {
        "estimate": spcor_result["estimate"][0, 1],
        "p_value": spcor_result["p_value"][0, 1],
        "statistic": spcor_result["statistic"][0, 1],
        "n": spcor_result["n"],
        "gp": spcor_result["gp"],
        "method": method,
    }


def get_all_vars_from_formula(formula) -> list:
    """Utility function to extract variables from a formula.

    Args:
        formula (str): A formula.

    Returns:
        list: A list of variables.

    Examples:
        >>> from spotpython.utils.stats import get_all_vars_from_formula
            get_all_vars_from_formula("y ~ x1 + x2")
                ['y', 'x1', 'x2']
            get_all_vars_from_formula("y ~ ")
                ['y']
    """
    # Split the formula into the dependent and independent variables
    dependent, independent = formula.split("~")
    # Strip whitespace and split the independent variables by '+'
    independent_vars = independent.strip().split("+") if independent.strip() else []
    # Combine the dependent variable with the independent variables
    return [dependent.strip()] + [var.strip() for var in independent_vars]


def fit_all_lm(basic, xlist, data, remove_na=True) -> dict:
    """Fit a linear regression model for all possible combinations of independent variables.

    Args:
        basic (str): The basic model formula.
        xlist (list): A list of independent variables.
        data (pandas.DataFrame): The data frame containing the variables.
        remove_na (bool): Whether to remove missing values from the data frame.

    Returns:
        dict: A dictionary containing the estimated coefficients, confidence intervals,
            p-values, AIC values, sample size, and the basic model formula.

    Examples:
        >>> from spotpython.utils.stats import fit_all_lm
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        >>>     'y': [1, 2, 3],
        >>>     'x1': [4, 5, 6],
        >>>     'x2': [7, 8, 9]
        >>> })
        >>> fit_all_lm("y ~ x1", ["x2"], data)
        {'estimate':   variables  estimate  conf_low  conf_high    p         aic  n
        0    basic  1.000000  1.000000   1.000000  0.0  0.000000  3
        1       x2  1.000000  1.000000   1.000000  0.0  0.000000  3}
    """
    # Prepare the data frame
    data = copy.deepcopy(data)
    data = data[get_all_vars_from_formula(basic) + xlist]
    if remove_na:
        data = data.dropna()
    print(data.head())
    # basic model
    mod_0 = ols(basic, data=data).fit()
    p = mod_0.pvalues.iloc[1]
    print(f"p-values: {p}")
    estimate = mod_0.params.iloc[1]
    print(f"estimate: {estimate}")
    conf_int = mod_0.conf_int().iloc[1]
    print(f"conf_int: {conf_int}")
    aic_value = mod_0.aic
    print(f"aic: {aic_value}")
    n = len(mod_0.resid)
    df_0 = pd.DataFrame([["basic", estimate, conf_int[0], conf_int[1], p, aic_value, n]], columns=["variables", "estimate", "conf_low", "conf_high", "p", "aic", "n"])

    # All combinations model
    comb_lst = list(itertools.chain.from_iterable(itertools.combinations(xlist, r) for r in range(1, len(xlist) + 1)))
    models = [ols(f"{basic} + {' + '.join(comb)}", data=data).fit() for comb in comb_lst]

    df_list = []
    for i, model in enumerate(models):
        p = model.pvalues.iloc[1]
        estimate = model.params.iloc[1]
        conf_int = model.conf_int().iloc[1]
        aic_value = model.aic
        n = len(model.resid)
        comb_str = ", ".join(comb_lst[i])
        df_list.append([comb_str, estimate, conf_int[0], conf_int[1], p, aic_value, n])

    df_coef = pd.DataFrame(df_list, columns=["variables", "estimate", "conf_low", "conf_high", "p", "aic", "n"])
    estimates = pd.concat([df_0, df_coef], ignore_index=True)
    return {"estimate": estimates, "xlist": xlist, "fun": "all_lm", "basic": basic, "family": "lm"}


def plot_coeff_vs_pvals(data, xlabels=None, xlim=(0, 1), xlab="p-value", ylim=None, ylab=None, xscale_log=True, yscale_log=False, title=None, show=True) -> None:
    """Plot the coefficient estimates from fit_all_lm against the corresponding p-values.

    Args:
        data (dict):
            A dictionary containing the estimated coefficients, p-values, and other information.
            Generated by the fit_all_lm function.
        xlabels (list):
            A list of x-axis labels.
        xlim (tuple):
            A tuple of the x-axis limits.
        xlab (str):
            The x-axis label.
        ylim (tuple):
            A tuple of the y-axis limits.
        ylab (str):
            The y-axis label.
        xscale_log (bool):
            Whether to use a log scale on the x-axis.
        yscale_log (bool):
            Whether to use a log scale on the y-axis.
        title (str):
            The plot title.
        show (bool):
            Whether to display the plot.

    Returns:
        None

    Notes:
        * Based on the R package 'allestimates' by Zhiqiang Wang, see https://cran.r-project.org/package=allestimates

    References:
        Wang, Z. (2007). Two Postestimation Commands for Assessing Confounding Effects in Epidemiological Studies. The Stata Journal, 7(2), 183-196. https://doi.org/10.1177/1536867X0700700203

    Examples:
        >>> from spotpython.utils.stats import plot_coeff_vs_pvals, fit_all_lm
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        >>>     'y': [1, 2, 3],
        >>>     'x1': [4, 5, 6],
        >>>     'x2': [7, 8, 9]
        >>> })
        >>> estimates = fit_all_lm("y ~ x1", ["x2"], data)
        >>> plot_coeff_vs_pvals(estimates)
    """
    data = copy.deepcopy(data)
    if xlabels is None:
        xlabels = [0, 0.001, 0.01, 0.05, 0.2, 0.5, 1]
    xbreaks = np.power(xlabels, np.log(0.5) / np.log(0.05))

    result_df = data["estimate"]
    if ylab is None:
        ylab = "Coefficient" if data["fun"] == "all_lm" else "Effect estimates"
    hline = 0 if data["fun"] == "all_lm" else 1

    result_df["p_value"] = np.power(result_df["p"], np.log(0.5) / np.log(0.05))
    if ylim is None:
        maxv = max(result_df["estimate"].max(), abs(result_df["estimate"].min()))
        ylim = (-maxv, maxv) if data["fun"] == "all_lm" else (1 / maxv, maxv)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=result_df, x="p_value", y="estimate")
    if xscale_log:
        plt.xscale("log")
    if yscale_log:
        plt.yscale("log")
    plt.xticks(ticks=xbreaks, labels=xlabels)
    plt.axvline(x=0.5, linestyle="--")
    plt.axhline(y=hline, linestyle="--")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if title:
        plt.title(title)
    plt.grid(True)
    if show:
        plt.show()


def plot_coeff_vs_pvals_by_included(data, xlabels=None, xlim=(0, 1), xlab="P value", ylim=None, ylab=None, yscale_log=False, title=None, grid=True, ncol=2, show=True) -> None:
    """
    Generates a panel of scatter plots with effect estimates of all possible models against p-values.
    Uses a dictionry generated by the fit_all_lm function.
    Each plot includes effect estimates from all models including a specific variable.

    Args:
        data (dict): A dictionary, generated by the fit_all_lm function, containing the following keys:
            - estimate (pd.DataFrame): A DataFrame containing the estimates.
            - xlist (list): A list of variables.
            - fun (str): The function name.
            - family (str): The family of the model.
        xlabels (list): A list of x-axis labels.
        xlim (tuple): The x-axis limits.
        xlab (str): The x-axis label.
        ylim (tuple): The y-axis limits.
        ylab (str): The y-axis label.
        yscale_log (bool): Whether to scale y-axis to log10. Default is False.
        title (str): The title of the plot.
        grid (bool): Whether to display gridlines. Default is True.
        ncol (int): Number of columns in the plot grid. Default is 2.

    Returns:
        None

    Notes:
        * Based on the R package 'allestimates' by Zhiqiang Wang, see https://cran.r-project.org/package=allestimates

    References:
        Wang, Z. (2007). Two Postestimation Commands for Assessing Confounding Effects in Epidemiological Studies. The Stata Journal, 7(2), 183-196. https://doi.org/10.1177/1536867X0700700203


    Examples:
        data = {
            "estimate": pd.DataFrame({
                "variables": ["Crude", "AL", "AM", "AN", "AO"],
                "estimate": [0.5, 0.6, 0.7, 0.8, 0.9],
                "conf_low": [0.1, 0.2, 0.3, 0.4, 0.5],
                "conf_high": [0.9, 1.0, 1.1, 1.2, 1.3],
                "p": [0.01, 0.02, 0.03, 0.04, 0.05],
                "aic": [100, 200, 300, 400, 500],
                "n": [10, 20, 30, 40, 50]
            }),
            "xlist": ["AL", "AM", "AN", "AO"],
            "fun": "all_lm"
        }
        plot_coeff_vs_pvals_by_included(data)
    """
    if xlabels is None:
        xlabels = [0, 0.001, 0.01, 0.05, 0.2, 0.5, 1]
    xbreaks = np.power(xlabels, np.log(0.5) / np.log(0.05))

    result_df = data["estimate"]
    if ylab is None:
        ylab = {"all_lm": "Coefficient", "poisson": "Rate ratio", "binomial": "Odds ratio"}.get(data.get("fun"), "Effect estimates")

    hline = 0 if data["fun"] == "all_lm" else 1

    result_df["p_value"] = np.power(result_df["p"], np.log(0.5) / np.log(0.05))
    if ylim is None:
        maxv = max(result_df["estimate"].max(), abs(result_df["estimate"].min()))
        if data["fun"] == "all_lm":
            ylim = (-maxv, maxv)
        else:
            ylim = (1 / maxv, maxv)

    # Create a DataFrame to mark inclusion of variables
    mark_df = pd.DataFrame({x: result_df["variables"].str.contains(x).astype(int) for x in data["xlist"]})
    df_scatter = pd.concat([result_df, mark_df], axis=1)

    # Melt the DataFrame for plotting
    df_long = df_scatter.melt(id_vars=["variables", "estimate", "conf_low", "conf_high", "p", "aic", "n", "p_value"], value_vars=data["xlist"], var_name="variable", value_name="inclusion")
    df_long["inclusion"] = df_long["inclusion"].apply(lambda x: "Included" if x > 0 else "Not included")

    # Plotting
    g = sns.FacetGrid(df_long, col="variable", hue="inclusion", palette={"Included": "blue", "Not included": "orange"}, col_wrap=ncol, height=4, sharex=False, sharey=False)
    g.map(sns.scatterplot, "p_value", "estimate")
    g.add_legend()
    for ax in g.axes.flat:
        ax.set_xticks(xbreaks)
        ax.set_xticklabels(xlabels)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axvline(x=0.5, linestyle="--", linewidth=1.5, color="black")  # Black dashed vertical line
        ax.axhline(y=hline, linestyle="--", linewidth=1.5, color="black")  # Black dashed horizontal line
        if grid:
            ax.grid(True)
    if yscale_log:
        g.set(yscale="log")
    g.set_axis_labels(xlab, ylab)
    g.set_titles("{col_name}")
    if title:
        plt.subplots_adjust(top=0.9)
        g.figure.suptitle(title)
    if show:
        plt.show()
