from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import matplotlib
from sklearn.base import BaseEstimator
from sklearn.metrics import PredictionErrorDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from spotpython.utils.convert import get_Xy_from_df
from typing import Any, Dict, List, Union
import pandas as pd


def plot_cv_predictions(model: Any, fun_control: Dict, show=True) -> None:
    """
    Plots cross-validated predictions for regression.

    Uses `sklearn.model_selection.cross_val_predict` together with
    `sklearn.metrics.PredictionErrorDisplay` to visualize prediction errors.
    It is based on the example from the scikit-learn documentation:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html#sphx-glr-download-auto-examples-model-selection-plot-cv-predict-py

    Args:
        model (Any):
            Sklearn model. The model to be used for cross-validation.
        fun_control (Dict):
            Dictionary containing the data and the target column.
        show (bool, optional):
            If True, the plot is shown. Defaults to True.

    Returns:
        (NoneType): None

    Examples:
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import LinearRegression
        >>> X, y = load_diabetes(return_X_y=True)
        >>> lr = LinearRegression()
        >>> plot_cv_predictions(lr, fun_control)
    """
    X_test, y_test = get_Xy_from_df(fun_control["test"], fun_control["target_column"])
    # cross_val_predict returns an array of the same size of y
    # where each entry is a prediction obtained by cross validation.
    y_pred = cross_val_predict(model, X_test, y_test, cv=10)
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        subsample=100,
        ax=axs[0],
        random_state=0,
    )
    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        subsample=100,
        ax=axs[1],
        random_state=0,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    fig.suptitle("Plotting cross-validated predictions")
    plt.tight_layout()
    if show:
        plt.show()


def plot_roc(
    model_list: List[BaseEstimator],
    fun_control: Dict[str, Union[str, pd.DataFrame]],
    alpha: float = 0.8,
    model_names: List[str] = None,
    show=True,
) -> None:
    """
    Plots ROC curves for a list of models using the Visualization API from scikit-learn.

    Args:
        model_list (List[BaseEstimator]):
            A list of scikit-learn models to plot ROC curves for.
        fun_control (Dict[str, Union[str, pd.DataFrame]]):
            A dictionary containing the train and test dataframes and the target column name.
        alpha (float, optional):
            The alpha value for the ROC curve. Defaults to 0.8.
        model_names (List[str], optional):
            A list of names for the models. Defaults to None.
        show (bool, optional):
            If True, the plot is shown. Defaults to True.

    Returns:
        (NoneType): None

    Examples:
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> iris = load_iris()
        >>> X_train = iris.data[:100]
        >>> y_train = iris.target[:100]
        >>> X_test = iris.data[100:]
        >>> y_test = iris.target[100:]
        >>> train_df = pd.DataFrame(X_train, columns=iris.feature_names)
        >>> train_df['target'] = y_train
        >>> test_df = pd.DataFrame(X_test, columns=iris.feature_names)
        >>> test_df['target'] = y_test
        >>> fun_control = {"train": train_df, "test": test_df, "target_column": "target"}
        >>> model_list = [LogisticRegression(), DecisionTreeClassifier()]
        >>> model_names = ["Logistic Regression", "Decision Tree"]
        >>> plot_roc(model_list, fun_control, model_names=model_names)
    """
    X_train, y_train = get_Xy_from_df(fun_control["train"], fun_control["target_column"])
    X_test, y_test = get_Xy_from_df(fun_control["test"], fun_control["target_column"])
    ax = plt.gca()
    for i, model in enumerate(model_list):
        model.fit(X_train, y_train)
        if model_names is not None:
            model_name = model_names[i]
        else:
            model_name = None
        y_pred = model.predict(X_test)
        RocCurveDisplay.from_predictions(y_test, y_pred, ax=ax, alpha=alpha, name=model_name)
    if show:
        plt.show()


def plot_roc_from_dataframes(
    df_list: List[pd.DataFrame],
    alpha: float = 0.8,
    model_names: List[str] = None,
    target_column: str = None,
    show: bool = True,
    title: str = "",
    tkagg: bool = False,
) -> None:
    """
    Plot ROC curve for a list of dataframes from model evaluations.

    Args:
        df_list:
            List of dataframes with results from models.
        alpha:
            Transparency of the plotted lines.
        model_names:
            List of model names.
        target_column:
            Name of the target column.
        show:
            If True, the plot is shown.
        title:
            Title of the plot.
        tkagg:
            If True, the TkAgg backend is used.
            Default is False.

    Returns:
        None

    Examples:
        >>> import pandas as pd
            from spotpython.plot.validation import plot_roc_from_dataframes
            df1 = pd.DataFrame({"y": [1, 0, 0, 1], "Prediction": [1,0,0,0]})
            df2 = pd.DataFrame({"y": [1, 0, 0, 1], "Prediction": [1,0,1,1]})
            df_list = [df1, df2]
            model_names = ["Model 1", "Model 2"]
            plot_roc_from_dataframes(df_list, model_names=model_names, target_column="y")

    """
    if tkagg:
        matplotlib.use("TkAgg")
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, df in enumerate(df_list):
        y_test = df[target_column]
        y_pred = df["Prediction"]
        if model_names is not None:
            model_name = model_names[i]
        else:
            model_name = None
        RocCurveDisplay.from_predictions(y_test, y_pred, ax=ax, alpha=alpha, name=model_name)
    # add a title to the plot
    ax.set_title(title)
    if show:
        plt.show()


def plot_confusion_matrix(
    model=None,
    fun_control=None,
    df=None,
    title=None,
    target_names=None,
    y_true_name=None,
    y_pred_name=None,
    show=False,
    ax=None,
):
    """
    Plotting a confusion matrix. If a model and the fun_control dictionary are passed,
    the confusion matrix is computed. If a dataframe is passed, the confusion matrix is
    computed from the dataframe. In this case, the names of the columns with the true and
    the predicted values must be specified. Default the dataframe is None.

    Args:
        model (Any, optional):
            Sklearn model. The model to be used for cross-validation. Defaults to None.
        fun_control (Dict, optional):
            Dictionary containing the data and the target column. Defaults to None.
        title (str, optional):
            Title of the plot. Defaults to None.
        df (pd.DataFrame, optional):
            Dataframe containing the predictions and the target column. Defaults to None.
        target_names (List[str], optional):
            List of target names. Defaults to None.
        y_true_name (str, optional):
            Name of the column with the true values if a dataframe is specified. Defaults to None.
        y_pred_name (str, optional):
            Name of the column with the predicted values if a dataframe is specified. Defaults to None.
        show (bool, optional):
            If True, the plot is shown. Defaults to False.
        ax (matplotlib.axes._subplots.AxesSubplot, optional):
            Axes to plot the confusion matrix. Defaults to None.

    Returns:
        (NoneType): None

    """
    if df is not None:
        # assign the column y_true_name from df to y_true
        y_true = df[y_true_name]
        # assign the column y_pred_name from df to y_pred
        y_pred = df[y_pred_name]
    else:
        X_train, y_train = get_Xy_from_df(fun_control["train"], fun_control["target_column"])
        X_test, y_true = get_Xy_from_df(fun_control["test"], fun_control["target_column"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, ax=ax, colorbar=False)
    if target_names is not None:
        ax.xaxis.set_ticklabels(target_names)
        ax.yaxis.set_ticklabels(target_names)
    if title is not None:
        _ = ax.set_title(title)
    if show:
        plt.show()


def plot_actual_vs_predicted(y_test, y_pred, title=None, show=True, filename=None) -> None:
    """Plot actual vs. predicted values.

    Args:
        y_test (np.ndarray):
            True values.
        y_pred (np.ndarray):
            Predicted values.
        title (str, optional):
            Title of the plot. Defaults to None.
        show (bool, optional):
            If True, the plot is shown. Defaults to True.
        filename (str, optional):
            Name of the file to save the plot. Defaults to None.

    Returns:
        (NoneType): None

    Examples:
        >>> from sklearn.datasets import load_diabetes
            from sklearn.linear_model import LinearRegression
            from spotpython.plot.validation import plot_actual_vs_predicted
            X, y = load_diabetes(return_X_y=True)
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            plot_actual_vs_predicted(y, y_pred)
    """
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        subsample=100,
        ax=axs[0],
        random_state=0,
        scatter_kwargs={"alpha": 0.5},
    )
    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        subsample=100,
        ax=axs[1],
        random_state=0,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
