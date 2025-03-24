import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np


def generate_mdi(X, y, feature_names=None, random_state=42) -> pd.DataFrame:
    """
    Generates a DataFrame with Gini importances from a RandomForestRegressor.

    Notes:
     There are two limitations of impurity-based feature importances:
        - impurity-based importances are biased towards high cardinality features;
        - impurity-based importances are computed on training set statistics
        and therefore do not reflect the ability of feature to be useful to
        make predictions that generalize to the test set. Permutation
        importances can mitigate the last limitation, because ti can be computed on the
        test set.

    Args:
        X (pd.DataFrame or np.ndarray): The feature set.
        y (pd.Series or np.ndarray): The target variable.
        feature_names (list, optional): List of feature names for labeling. Defaults to None.
        random_state (int, optional): Random state for the RandomForestRegressor. Defaults to 42.

    Returns:
        pd.DataFrame: DataFrame with 'Feature' and 'Importance' columns.

    Examples:
        >>> from spotpython.plot.importance import generate_mdi
        >>> import pandas as pd
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        >>> X_df = pd.DataFrame(X)
        >>> y_series = pd.Series(y)
        >>> result = generate_mdi(X_df, y_series)
        >>> print(result)

    """
    # Convert X and y to pandas DataFrames if they are not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(np.ravel(y))  # Use np.ravel instead of flatten

    # Train a Random Forest Regressor
    rf = RandomForestRegressor(random_state=random_state)
    rf.fit(X, y)

    # Get feature importances
    importances = rf.feature_importances_

    # Create a DataFrame
    if feature_names is None:
        df_mdi = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    else:
        df_mdi = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_mdi = df_mdi.sort_values("Importance", ascending=False).reset_index(drop=True)

    return df_mdi


def generate_imp(X_train, X_test, y_train, y_test, random_state=42, n_repeats=10, use_test=True) -> permutation_importance:
    """
    Generates permutation importances from a RandomForestRegressor.

    Args:
        X_train (pd.DataFrame or np.ndarray): The training feature set.
        X_test (pd.DataFrame or np.ndarray): The test feature set.
        y_train (pd.Series or np.ndarray): The training target variable.
        y_test (pd.Series or np.ndarray): The test target variable.
        random_state (int, optional): Random state for the RandomForestRegressor. Defaults to 42.
        n_repeats (int, optional): Number of repeats for permutation importance. Defaults to 10.
        use_test (bool, optional): If True, computes permutation importance on the test set. If False, uses the training set. Defaults to True.

    Returns:
        permutation_importance: Permutation importances object.

    Examples:
        >>> from spotpython.plot.importance import generate_imp
        >>> import pandas as pd
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        >>> X_train, X_test = X[:80], X[80:]
        >>> y_train, y_test = y[:80], y[80:]
        >>> X_train_df = pd.DataFrame(X_train)
        >>> X_test_df = pd.DataFrame(X_test)
        >>> y_train_series = pd.Series(y_train)
        >>> y_test_series = pd.Series(y_test)
        >>> perm_imp = generate_imp(X_train_df, X_test_df, y_train_series, y_test_series)
        >>> print(perm_imp)
    """
    # Convert inputs to pandas DataFrames/Series if they are not already
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(np.ravel(y_train))  # Use np.ravel instead of flatten
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(np.ravel(y_test))  # Use np.ravel instead of flatten

    # Train a Random Forest Regressor
    rf = RandomForestRegressor(random_state=random_state)
    rf.fit(X_train, y_train)

    # Select the dataset for permutation importance
    X_eval = X_test if use_test else X_train
    y_eval = y_test if use_test else y_train

    # Calculate permutation importances
    perm_imp = permutation_importance(rf, X_eval, y_eval, n_repeats=n_repeats, random_state=random_state)

    return perm_imp


def plot_importances(df_mdi, perm_imp, X_test, target_name=None, feature_names=None, k=10, show=True) -> None:
    """
    Plots the impurity-based and permutation-based feature importances for a given classifier.

    Args:
        df_mdi (pd.DataFrame):
            DataFrame with Gini importances.
        perm_imp (object):
            Permutation importances object.
        X_test (pd.DataFrame):
            The test feature set for permutation importance.
        target_name (str, optional):
            Name of the target variable for labeling. Defaults to None.
        feature_names (list, optional):
            List of feature names for labeling. Defaults to None.
        k (int, optional):
            Number of top features to display based on importance. Default is 10.
        show (bool, optional):
            If True, displays the plot immediately. Default is True.

    Returns:
        None

    Examples:
        >>> from spotpython.plot.importance import generate_mdi, generate_imp, plot_importances
        >>> import pandas as pd
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        >>> X_train, X_test = X[:80], X[80:]
        >>> y_train, y_test = y[:80], y[80:]
        >>> X_train_df = pd.DataFrame(X_train)
        >>> X_test_df = pd.DataFrame(X_test)
        >>> y_train_series = pd.Series(y_train)
        >>> y_test_series = pd.Series(y_test)
        >>> df_mdi = generate_mdi(X_train_df, y_train_series)
        >>> perm_imp = generate_imp(X_train_df, X_test_df, y_train_series, y_test_series)
        >>> plot_importances(df_mdi, perm_imp, X_test_df)

    """

    # Plot impurity-based importances for top-k features
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    sorted_mdi_importances = df_mdi.set_index("Feature")["Importance"]
    sorted_mdi_importances[:k].sort_values().plot.barh(ax=ax1)
    ax1.set_xlabel("Gini importance")
    if target_name:
        ax1.set_title(f"Impurity-based feature importances for target: {target_name}")
    else:
        ax1.set_title("Impurity-based feature importances")

    # Ensure X_test is a DataFrame
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    perm_sorted_idx = perm_imp.importances_mean.argsort()[-k:]
    if feature_names is not None:
        ax2.boxplot(perm_imp.importances[perm_sorted_idx].T, vert=False, labels=np.array(feature_names)[perm_sorted_idx])
    else:
        ax2.boxplot(perm_imp.importances[perm_sorted_idx].T, vert=False, labels=X_test.columns[perm_sorted_idx])
    ax2.axvline(x=0, color="k", linestyle="--")
    if target_name:
        ax2.set_xlabel(f"Decrease in mse for target: {target_name}")
    else:
        ax2.set_xlabel("Decrease in mse")
    ax2.set_title("Permutation-based feature importances")

    # fig.suptitle("Impurity-based vs. permutation importances")
    fig.tight_layout()
    if show:
        plt.show()
