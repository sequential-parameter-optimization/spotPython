"""
This module implements pytorch lightning neural networks for handling regression tasks.

"""

from .nn_resnet_regressor import NNResNetRegressor
from .nn_transformer_regressor import NNTransformerRegressor
from .nn_linear_regressor import NNLinearRegressor
from .netlightregression import NetLightRegression
from .nn_condnet_regressor import NNCondNetRegressor
from .nn_many_to_many_rnn_regressor import ManyToManyRNNRegressor, ManyToManyRNN

__all__ = [
    "NNLinearRegressor",
    "NetLightRegression",
    "NNResNetRegressor",
    "NNTransformerRegressor",
    "NNCondNetRegressor",
    "ManyToManyRNNRegressor",
    "ManyToManyRNN",
]
