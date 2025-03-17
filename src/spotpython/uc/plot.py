import numpy as np
import matplotlib.pyplot as plt


def plot_predictionintervals(
    y_train,
    y_train_pred,
    y_train_pred_low,
    y_train_pred_high,
    y_test,
    y_test_pred,
    y_test_pred_low,
    y_test_pred_high,
    suptitle: str,
) -> None:
    """
    Plots prediction intervals for training and testing data.
    This function generates four subplots arranged in a 2x2 grid:
    1. True vs predicted values with error bars representing prediction intervals.
    2. Prediction interval width vs true values.
    3. Ordered prediction interval widths for both training and testing data.
    4. Histograms of the interval widths for training and testing data.

    Args:
        y_train (array-like): True values for the training set.
        y_train_pred (array-like): Predicted values for the training set.
        y_train_pred_low (array-like): Lower bounds of prediction intervals for the training set.
        y_train_pred_high (array-like): Upper bounds of prediction intervals for the training set.
        y_test (array-like): True values for the testing set.
        y_test_pred (array-like): Predicted values for the testing set.
        y_test_pred_low (array-like): Lower bounds of prediction intervals for the testing set.
        y_test_pred_high (array-like): Upper bounds of prediction intervals for the testing set.
        suptitle (str): The title for the entire figure.

    Returns:
        None: The function displays the plots but does not return any value.

    Notes:
        - The first subplot compares true and predicted values with error bars for both training
          and testing data.
        - The second subplot visualizes the width of prediction intervals as a function of true values.
        - The third subplot orders the prediction interval widths and displays them for both
          training and testing data.
        - The fourth subplot shows histograms of the interval widths for training and testing data.

    References:
        Function adapted from: https://github.com/scikit-learn-contrib/MAPIE/blob/master/notebooks/regression/exoplanets.ipynb

    Examples:
        >>> import numpy as np
        >>> from spotpython.uc.plot import plot_predictionintervals
        >>> y_train = np.array([1, 2, 3, 4, 5])
        >>> y_train_pred = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        >>> y_train_pred_low = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y_train_pred_high = np.array([1.2, 2.4, 3.6, 4.8, 6.0])
        >>> y_test = np.array([6, 7, 8])
        >>> y_test_pred = np.array([6.1, 7.2, 8.3])
        >>> y_test_pred_low = np.array([6.0, 7.0, 8.0])
        >>> y_test_pred_high = np.array([6.2, 7.4, 8.6])
        >>> suptitle = "Prediction Intervals"
        >>> plot_predictionintervals(y_train, y_train_pred, y_train_pred_low, y_train_pred_high, y_test, y_test_pred, y_test_pred_low, y_test_pred_high, suptitle)
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    ax1.errorbar(
        x=y_train,
        y=y_train_pred,
        yerr=(y_train_pred - y_train_pred_low, y_train_pred_high - y_train_pred),
        alpha=0.8,
        label="train",
        fmt=".",
    )
    ax1.errorbar(
        x=y_test,
        y=y_test_pred,
        yerr=(y_test_pred - y_test_pred_low, y_test_pred_high - y_test_pred),
        alpha=0.8,
        label="test",
        fmt=".",
    )
    ax1.plot(
        [y_train.min(), y_train.max()],
        [y_train.min(), y_train.max()],
        color="gray",
        alpha=0.5,
    )
    ax1.set_xlabel("True values", fontsize=12)
    ax1.set_ylabel("Predicted values", fontsize=12)
    ax1.legend()
    ax1.set_title("True vs predicted values")

    ax2.scatter(x=y_train, y=y_train_pred_high - y_train_pred_low, alpha=0.8, label="train", marker=".")
    ax2.scatter(x=y_test, y=y_test_pred_high - y_test_pred_low, alpha=0.8, label="test", marker=".")
    ax2.set_xlabel("True values", fontsize=12)
    ax2.set_ylabel("Interval width", fontsize=12)
    ax2.set_xscale("linear")
    ax2.set_ylim([0, np.max(y_test_pred_high - y_test_pred_low) * 1.1])
    ax2.legend()
    ax2.set_title("Prediction interval width vs true values")

    std_all = np.concatenate([y_train_pred_high - y_train_pred_low, y_test_pred_high - y_test_pred_low])
    type_all = np.array(["train"] * len(y_train) + ["test"] * len(y_test))
    x_all = np.arange(len(std_all))
    order_all = np.argsort(std_all)
    std_order = std_all[order_all]
    type_order = type_all[order_all]
    ax3.scatter(
        x=x_all[type_order == "train"],
        y=std_order[type_order == "train"],
        alpha=0.8,
        label="train",
        marker=".",
    )
    ax3.scatter(
        x=x_all[type_order == "test"],
        y=std_order[type_order == "test"],
        alpha=0.8,
        label="test",
        marker=".",
    )
    ax3.set_xlabel("Order", fontsize=12)
    ax3.set_ylabel("Interval width", fontsize=12)
    ax3.legend()
    ax3.set_title("Ordered prediction interval width")

    ax4.hist(y_train_pred_high - y_train_pred_low, alpha=0.5, label="train")
    ax4.hist(y_test_pred_high - y_test_pred_low, alpha=0.5, label="test")
    ax4.set_xlabel("Interval width", fontsize=12)
    ax4.set_ylabel("Frequency", fontsize=12)
    ax4.legend()
    ax4.set_title("Histogram of interval widths")

    plt.suptitle(suptitle, size=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show()
