"""
This module implements pytorch lightning neural networks for handling regression tasks.

"""

from .nn_linear_regressor import NNLinearRegressor
from .netlightregression import NetLightRegression

__all__ = ["NNLinearRegressor", "NetLightRegression"]
