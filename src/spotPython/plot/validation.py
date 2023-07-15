from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import PredictionErrorDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from spotPython.utils.convert import get_Xy_from_df
from typing import Any, Dict, List, Union
import pandas as pd


def plot_cv_predictions(model: Any, fun_control: Dict) -> None:
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
    plt.show()


def plot_roc(
    model_list: List[BaseEstimator],
    fun_control: Dict[str, Union[str, pd.DataFrame]],
    alpha: float = 0.8,
    model_names: List[str] = None,
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
    plt.show()


def plot_confusion_matrix(model, fun_control, target_names=None, title=None):
    """
    Plotting a confusion matrix
    """
    X_train, y_train = get_Xy_from_df(fun_control["train"], fun_control["target_column"])
    X_test, y_test = get_Xy_from_df(fun_control["test"], fun_control["target_column"])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
    if target_names is not None:
        ax.xaxis.set_ticklabels(target_names)
        ax.yaxis.set_ticklabels(target_names)
    if title is not None:
        _ = ax.set_title(title)


def plot_actual_vs_predicted(y_test, y_pred, title=None):
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
    plt.show()
