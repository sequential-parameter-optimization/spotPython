import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split


def randorient(k, p, xi, seed=None) -> np.ndarray:
    """Generates a random orientation of a sampling matrix.
    This function creates a random sampling matrix for a given number of
    dimensions (k), number of levels (p), and step length (xi). The
    resulting matrix is used for screening designs in the context of
    experimental design.

    Args:
        k (int): Number of dimensions.
        p (int): Number of levels.
        xi (float): Step length.
        seed (int, optional): Seed for the random number generator.
            Defaults to None.

    Returns:
        np.ndarray: A random sampling matrix of shape (k+1, k).

    Example:
        >>> randorient(k=2, p=3, xi=0.5)
        array([[0. , 0. ],
               [0.5, 0.5],
               [1. , 1. ]])
    """
    # Initialize random number generator with the provided seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Step length
    Delta = xi / (p - 1)

    m = k + 1

    # A truncated p-level grid in one dimension
    xs = np.arange(0, 1 - Delta, 1 / (p - 1))
    xsl = len(xs)
    if xsl < 1:
        print(f"xi = {xi}.")
        print(f"p = {p}.")
        print(f"Delta = {Delta}.")
        print(f"p - 1 = {p - 1}.")
        raise ValueError(f"The number of levels xsl is {xsl}, but it must be greater than 0.")

    # Basic sampling matrix
    B = np.vstack((np.zeros((1, k)), np.tril(np.ones((k, k)))))

    # Randomization

    # Matrix with +1s and -1s on the diagonal with equal probability
    Dstar = np.diag(2 * rng.integers(0, 2, size=k) - 1)

    # Random base value
    xstar = xs[rng.integers(0, xsl, size=k)]

    # Permutation matrix
    Pstar = np.zeros((k, k))
    rp = rng.permutation(k)
    for i in range(k):
        Pstar[i, rp[i]] = 1

    # A random orientation of the sampling matrix
    Bstar = (np.ones((m, 1)) @ xstar.reshape(1, -1) + (Delta / 2) * ((2 * B - np.ones((m, k))) @ Dstar + np.ones((m, k)))) @ Pstar

    return Bstar


def screeningplan(k, p, xi, r):
    # Empty list to accumulate screening plan rows
    X = []

    for i in range(r):
        X.append(randorient(k, p, xi))

    # Concatenate list of arrays into a single array
    X = np.vstack(X)

    return X


def _screening(X, fun, xi, p, labels, bounds=None) -> tuple:
    """Helper function to calculate elementary effects for a screening design.

    Args:
        X (np.ndarray): The screening plan matrix, typically structured
            within a [0,1]^k box.
        fun (object): The objective function to evaluate at each
            design point in the screening plan.
        xi (float): The elementary effect step length factor.
        p (int): Number of discrete levels along each dimension.
        labels (list of str): A list of variable names corresponding to
            the design variables.
        bounds (np.ndarray): A 2xk matrix where the first row contains
            lower bounds and the second row contains upper bounds for
            each variable.

    Returns:
        tuple: A tuple containing two arrays:
            - sm: The mean of the elementary effects for each variable.
            - ssd: The standard deviation of the elementary effects for
            each variable.

    Examples:
        >>> import numpy as np
            from spotpython.utils.effects import screening, screeningplan
            from spotpython.fun.objectivefunctions import Analytical
            fun = Analytical()
            k = 10
            p = 10
            xi = 1
            r = 25
            X = screeningplan(k=k, p=p, xi=xi, r=r)  # shape (r x (k+1), k)
            # Provide real-world bounds from the wing weight docs (2 x 10).
            value_range = np.array([
                [150, 220,   6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025],
                [200, 300,  10,  10, 45, 1.0, 0.18, 6.0, 2500, 0.08 ],
            ])
            labels = [
                "S_W", "W_fw", "A", "Lambda",
                "q",   "lambda", "tc", "N_z",
                "W_dg", "W_p"
            ]
            screening(
                X=X,
                fun=fun.fun_wingwt,
                bounds=value_range,
                xi=xi,
                p=p,
                labels=labels,
                print=False,
            )
    """
    k = X.shape[1]
    r = X.shape[0] // (k + 1)

    # Scale each design point
    t = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if bounds is not None:
            X[i, :] = bounds[0, :] + X[i, :] * (bounds[1, :] - bounds[0, :])
        t[i] = fun(X[i, :])

    # Elementary effects
    F = np.zeros((k, r))
    for i in range(r):
        for j in range(i * (k + 1), i * (k + 1) + k):
            idx = np.where(X[j, :] - X[j + 1, :] != 0)[0][0]
            F[idx, i] = (t[j + 1] - t[j]) / (xi / (p - 1))

    # Statistical measures (divide by n)
    ssd = np.std(F, axis=1, ddof=0)
    sm = np.mean(F, axis=1)
    return sm, ssd


def screening_print(X, fun, xi, p, labels, bounds=None) -> pd.DataFrame:
    """Generates a DataFrame with elementary effect screening metrics.

    This function calculates the mean and standard deviation of the
    elementary effects for a given set of design variables and returns
    the results as a Pandas DataFrame.

    Args:
        X (np.ndarray): The screening plan matrix, typically structured
            within a [0,1]^k box.
        fun (object): The objective function to evaluate at each
            design point in the screening plan.
        xi (float): The elementary effect step length factor.
        p (int): Number of discrete levels along each dimension.
        labels (list of str): A list of variable names corresponding to
            the design variables.
        bounds (np.ndarray): A 2xk matrix where the first row contains
            lower bounds and the second row contains upper bounds for
            each variable.

    Returns:
        pd.DataFrame: A DataFrame containing three columns:
            - 'varname': The name of each variable.
            - 'mean': The mean of the elementary effects for each variable.
            - 'sd': The standard deviation of the elementary effects for
            each variable.
        or None: If print is set to False, a plot of the results is
            generated instead of returning a DataFrame.

    Examples:
        >>> import numpy as np
            from spotpython.utils.effects import screening, screeningplan
            from spotpython.fun.objectivefunctions import Analytical
            fun = Analytical()
            k = 10
            p = 10
            xi = 1
            r = 25
            X = screeningplan(k=k, p=p, xi=xi, r=r)  # shape (r x (k+1), k)
            # Provide real-world bounds from the wing weight docs (2 x 10).
            value_range = np.array([
                [150, 220,   6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025],
                [200, 300,  10,  10, 45, 1.0, 0.18, 6.0, 2500, 0.08 ],
            ])
            labels = [
                "S_W", "W_fw", "A", "Lambda",
                "q",   "lambda", "tc", "N_z",
                "W_dg", "W_p"
            ]
            screening(
                X=X,
                fun=fun.fun_wingwt,
                bounds=value_range,
                xi=xi,
                p=p,
                labels=labels,
                print=False,
            )
    """
    sm, ssd = _screening(X=X, fun=fun, xi=xi, p=p, labels=labels, bounds=bounds)
    idx = np.argsort(-np.abs(sm))
    sorted_labels = [labels[i] for i in idx]
    sm = sm[idx]
    ssd = ssd[idx]
    df = pd.DataFrame({"varname": sorted_labels, "mean": sm, "sd": ssd})
    return df


def screening_plot(X, fun, xi, p, labels, bounds=None, show=True) -> None:
    """Generates a plot with elementary effect screening metrics.

    This function calculates the mean and standard deviation of the
    elementary effects for a given set of design variables and plots
    the results.

    Args:
        X (np.ndarray):
            The screening plan matrix, typically structured within a [0,1]^k box.
        fun (object):
            The objective function to evaluate at each design point in the screening plan.
        xi (float):
            The elementary effect step length factor.
        p (int):
            Number of discrete levels along each dimension.
        labels (list of str):
            A list of variable names corresponding to the design variables.
        bounds (np.ndarray):
            A 2xk matrix where the first row contains lower bounds and
            the second row contains upper bounds for each variable.
        show (bool):
            If True, the plot is displayed. Defaults to True.

    Returns:
        None: The function generates a plot of the results.

    Examples:
        >>> import numpy as np
            from spotpython.utils.effects import screening, screeningplan
            from spotpython.fun.objectivefunctions import Analytical
            fun = Analytical()
            k = 10
            p = 10
            xi = 1
            r = 25
            X = screeningplan(k=k, p=p, xi=xi, r=r)  # shape (r x (k+1), k)
            # Provide real-world bounds from the wing weight docs (2 x 10).
            value_range = np.array([
                [150, 220,   6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025],
                [200, 300,  10,  10, 45, 1.0, 0.18, 6.0, 2500, 0.08 ],
            ])
            labels = [
                "S_W", "W_fw", "A", "Lambda",
                "q",   "lambda", "tc", "N_z",
                "W_dg", "W_p"
            ]
            screening(
                X=X,
                fun=fun.fun_wingwt,
                bounds=value_range,
                xi=xi,
                p=p,
                labels=labels,
                print=False,
            )
    """
    k = X.shape[1]
    sm, ssd = _screening(X=X, fun=fun, xi=xi, p=p, labels=labels, bounds=bounds)
    plt.figure()
    for i in range(k):
        plt.text(sm[i], ssd[i], labels[i], fontsize=10)
    plt.axis([min(sm), 1.1 * max(sm), min(ssd), 1.1 * max(ssd)])
    plt.xlabel("Sample means")
    plt.ylabel("Sample standard deviations")
    plt.gca().tick_params(labelsize=10)
    plt.grid(True)
    if show:
        plt.show()


def plot_all_partial_dependence(df, df_target, model="GradientBoostingRegressor", nrows=5, ncols=6, figsize=(20, 15), title="") -> None:
    """
    Generates Partial Dependence Plots (PDPs) for every feature in a DataFrame against a target variable,
    arranged in a grid.

    Args:
        df (pd.DataFrame): DataFrame containing the features.
        df_target (pd.Series): Series containing the target variable.
        model (str, optional): Name of the model class to use (e.g., "GradientBoostingRegressor").
                               Defaults to "GradientBoostingRegressor".
        nrows (int, optional): Number of rows in the grid of subplots. Defaults to 5.
        ncols (int, optional): Number of columns in the grid of subplots. Defaults to 6.
        figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (20, 15).
        title (str, optional): Title for the subplots. Defaults to "".

    Returns:
        None

    Examples:
        >>> form spotpython.utils.effects import plot_all_partial_dependence
        >>> from sklearn.datasets import load_boston
        >>> import pandas as pd
        >>> data = load_boston()
        >>> df = pd.DataFrame(data.data, columns=data.feature_names)
        >>> df_target = pd.Series(data.target, name="target")
        >>> plot_all_partial_dependence(df, df_target, model="GradientBoostingRegressor", nrows=5, ncols=6, figsize=(20, 15))

    """

    # Separate features and target
    X = df
    y = df_target  # Target variable is now a Series

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the model
    if model == "GradientBoostingRegressor":
        gb_model = GradientBoostingRegressor(random_state=42)
    elif model == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor

        gb_model = RandomForestRegressor(random_state=42)
    elif model == "DecisionTreeRegressor":
        from sklearn.tree import DecisionTreeRegressor

        gb_model = DecisionTreeRegressor(random_state=42)
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Train model
    gb_model.fit(X_train, y_train)

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Generate PDP for each feature
    features = X.columns
    for i, feature in enumerate(features):
        ax = axes[i]  # Select the axis for the current feature
        PartialDependenceDisplay.from_estimator(gb_model, X_train, [feature], ax=ax)
        ax.set_title(title)  # Set the title of the subplot to the feature name

    # Remove empty subplots if the number of features is less than nrows * ncols
    for i in range(len(features), nrows * ncols):
        fig.delaxes(axes[i])

    plt.tight_layout()  # Adjust subplot parameters for a tight layout
    plt.show()
