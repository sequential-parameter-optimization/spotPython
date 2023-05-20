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
from spotPython.utils.convert import series_to_array


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
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
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
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
    Parameters
    ----------
    estimator : sklearn estimator
                The estimator to be used for prediction.
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    y_pred = estimator.predict_proba(X)
    score = mapk_score(y, y_pred, k=3)
    return score
