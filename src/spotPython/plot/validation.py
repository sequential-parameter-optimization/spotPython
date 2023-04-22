from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay
from spotPython.utils.convert import get_Xy_from_df
from sklearn.metrics import RocCurveDisplay


def plot_cv_predictions(model, fun_control):
    """
    Regression: Plotting Cross-Validated Predictions.
    Uses
    :func:`~sklearn.model_selection.cross_val_predict` together with
    :class:`~sklearn.metrics.PredictionErrorDisplay` to visualize prediction
    errors. It is based on the example from the scikit-learn documentation:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html#sphx-glr-download-auto-examples-model-selection-plot-cv-predict-py

    Parameters:
        model: sklearn model. The model to be used for cross-validation.
        fun_control: dict. The dictionary containing the data and the target column.
    Returns:
        None.
    Example:
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


def plot_roc(model_list, fun_control, alpha=0.8, model_names=None):
    """
    ================================
    ROC Curve with Visualization API
    ================================
    Scikit-learn defines a simple API for creating visualizations for machine
    learning. The key features of this API is to allow for quick plotting and
    visual adjustments without recalculation. In this example, we will demonstrate
    how to use the visualization API by comparing ROC curves.
    """
    X_test, y_test = get_Xy_from_df(fun_control["test"], fun_control["target_column"])
    ax = plt.gca()
    for i, model in enumerate(model_list):
        model.fit(X_test, y_test)
        if model_names is not None:
            model_name = model_names[i]
        else:
            model_name = None
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, alpha=alpha, name=model_name)
    plt.show()
