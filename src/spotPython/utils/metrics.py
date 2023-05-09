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


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two arrays of
    items.
    Parameters
    ----------
    actual : array
        An array of elements that are to be predicted (order doesn't matter)
    predicted : array
        An array of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
        The average precision at k over the input arrays
    """
    print("k:", k)
    if predicted.shape[0] > k:
        predicted = predicted[:k]
    print("predicted:", predicted)
    print("actual:", actual)
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        print("res:", 0)
        print("---------------------")
        return 0.0

    res = score / min(len(actual), k)
    print("res:", res)
    print("---------------------")
    return res


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average precision at k between two arrays
    of arrays of items.
    Parameters
    ----------
    actual : array
        An array of arrays of elements that are to be predicted
        (order doesn't matter in the arrays)
    predicted : array
        An array of arrays of predicted elements
        (order matters in the arrays)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
        The mean average precision at k over the input arrays
    """
    predicted = np.array(predicted).reshape(-1, 1)
    actual = np.array(actual).reshape(-1, 1)
    print("mapk: predicted:", predicted)
    print("mapk: actual:", actual)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
