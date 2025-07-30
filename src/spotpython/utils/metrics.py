# Sourced from the ml_metrics package at
# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
# Copyright (c) 2012, Ben Hamner
# Author: Ben Hamner (ben@benhamner.com)
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from spotpython.utils.convert import series_to_array


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.

    Args:
        actual (list): A list of elements that are to be predicted (order doesn't matter)
        predicted (list): A list of predicted elements (order does matter)
        k (int): The maximum number of predicted elements

    Returns:
        score (float): The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average precision at k between two lists
    of lists of items.

    Args:
        actual (list): A list of lists of elements that are to be predicted
            (order doesn't matter in the lists)
        predicted (list): A list of lists of predicted elements
            (order matters in the lists)
        k (int): The maximum number of predicted elements

    Returns:
        score (float): The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def mapk_score(y_true, y_pred, k=3):
    """Wrapper for mapk func using numpy arrays

     Args:
            y_true (np.array): array of true values
            y_pred (np.array): array of predicted values
            k (int): number of predictions

    Returns:
            score (float): mean average precision at k

    Examples:
            >>> y_true = np.array([0, 1, 2, 2])
            >>> y_pred = np.array([[0.5, 0.2, 0.2],  # 0 is in top 2
                     [0.3, 0.4, 0.2],  # 1 is in top 2
                     [0.2, 0.4, 0.3],  # 2 is in top 2
                     [0.7, 0.2, 0.1]]) # 2 isn't in top 2
            >>> mapk_score(y_true, y_pred, k=1)
            0.25
            >>> mapk_score(y_true, y_pred, k=2)
            0.375
            >>> mapk_score(y_true, y_pred, k=3)
            0.4583333333333333
            >>> mapk_score(y_true, y_pred, k=4)
            0.4583333333333333
    """
    y_true = series_to_array(y_true)
    sorted_prediction_ids = np.argsort(-y_pred, axis=1)
    top_k_prediction_ids = sorted_prediction_ids[:, :k]
    score = mapk(y_true.reshape(-1, 1), top_k_prediction_ids, k=k)
    return score


def mapk_scorer(estimator, X, y):
    """
    Scorer for mean average precision at k.
    This function computes the mean average precision at k between two lists
    of lists of items.

    Args:
        estimator (sklearn estimator): The estimator to be used for prediction.
        X (array-like of shape (n_samples, n_features)): The input samples.
        y (array-like of shape (n_samples,)): The target values.

    Returns:
        score (float): The mean average precision at k over the input lists
    """
    y_pred = estimator.predict_proba(X)
    score = mapk_score(y, y_pred, k=3)
    return score


def get_metric_sign(metric_name):
    """Returns the sign of a metric.

    Args:
        metric_name (str):
            The name of the metric. Can be one of the following:
                - "accuracy_score"
                - "cohen_kappa_score"
                - "f1_score"
                - "hamming_loss"
                - "hinge_loss"
                -"jaccard_score"
                - "matthews_corrcoef"
                - "precision_score"
                - "recall_score"
                - "roc_auc_score"
                - "zero_one_loss"

    Returns:
        sign (float): The sign of the metric. -1 for max, +1 for min.

    Raises:
        ValueError: If the metric is not found.

    Examples:
        >>> from spotpython.metrics import get_metric_sign
        >>> get_metric_sign("accuracy_score")
        -1
        >>> get_metric_sign("hamming_loss")
        +1

    """
    if metric_name in [
        "accuracy_score",
        "cohen_kappa_score",
        "f1_score",
        "jaccard_score",
        "matthews_corrcoef",
        "precision_score",
        "recall_score",
        "roc_auc_score",
        "explained_variance_score",
        "r2_score",
        "d2_absolute_error_score",
        "d2_pinball_score",
        "d2_tweedie_score",
    ]:
        return -1
    elif metric_name in [
        "hamming_loss",
        "hinge_loss",
        "zero_one_loss",
        "max_error",
        "mean_absolute_error",
        "mean_squared_error",
        "root_mean_squared_error",
        "mean_squared_log_error",
        "root_mean_squared_log_error",
        "median_absolute_error",
        "mean_poisson_deviance",
        "mean_gamma_deviance",
        "mean_absolute_percentage_error",
    ]:
        return +1
    else:
        raise ValueError(f"Metric '{metric_name}' not found.")


def calculate_xai_consistency(attributions) -> float:
    """Calculate the consistency between different XAI methods.
    Computes the pairwise correlation between different XAI methods' attributions
    and returns their mean correlation as a measure of consistency. A higher value
    indicates greater agreement between different XAI methods.

    Args:
        attributions (np.ndarray): Array of shape (n_methods, n_features) containing
            feature importance scores from different XAI methods. Each row represents
            a different XAI method's attributions, and each column represents a feature.

    Returns:
        float: Mean correlation between XAI methods, ranging from -1 to 1.
            - 1: Perfect consistency between methods
            - 0: No consistency between methods
            - -1: Perfect negative consistency between methods

    Examples:
        >>> import numpy as np
        >>> # Three XAI methods' attributions for four features
        >>> attributions = np.array([
        ...     [0.1, 0.2, 0.3, 0.4],  # Method 1
        ...     [0.2, 0.3, 0.4, 0.5],  # Method 2
        ...     [0.0, 0.1, 0.2, 0.3]   # Method 3
        ... ])
        >>> consistency = calculate_xai_consistency(attributions)
        >>> print(f"XAI Consistency: {consistency:.2f}")
        Attribution Correlation Matrix:
        [[ 1.    0.97  0.98]
         [ 0.97  1.    0.99]
         [ 0.98  0.99  1.  ]]
        XAI Consistency: 0.98

    Note:
        The correlation matrix is computed using numpy's corrcoef function, which
        calculates Pearson correlation coefficients. Only the upper triangle of
        the correlation matrix is used to avoid counting correlations twice.
    """
    global_attr_np = np.array(attributions)
    corr_matrix = np.corrcoef(global_attr_np)
    print("Attribution Correlation Matrix:")
    print(corr_matrix)

    # Calculate the mean of the upper triangle of the correlation matrix
    upper_triangle_indices = np.triu_indices_from(corr_matrix, k=1)
    upper_triangle_values = corr_matrix[upper_triangle_indices]
    result_xai = upper_triangle_values.mean()
    print("XAI Consistency (mean of upper triangle of correlation matrix):")
    print(result_xai)
    return result_xai
