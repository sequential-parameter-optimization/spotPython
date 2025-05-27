"""
This module implements pytorch lightning neural networks for handling regression tasks.

"""

from .nn_resnet_regressor import NNResNetRegressor
from .nn_transformer_regressor import NNTransformerRegressor
from .nn_linear_regressor import NNLinearRegressor
from .netlightregression import NetLightRegression
from .nn_funnel_regressor import NNFunnelRegressor
from .nn_condnet_regressor import NNCondNetRegressor
from .nn_many_to_many_rnn_regressor import ManyToManyRNNRegressor, ManyToManyRNN
from .nn_many_to_many_gru_regressor import ManyToManyGRURegressor
from .nn_many_to_many_lstm_regressor import ManyToManyLSTMRegressor

__all__ = [
    "NNLinearRegressor",
    "NetLightRegression",
    "NNFunnelRegressor",
    "NNResNetRegressor",
    "NNTransformerRegressor",
    "NNCondNetRegressor",
    "ManyToManyRNNRegressor",
    "ManyToManyRNN",
    "ManyToManyGRURegressor",
    "ManyToManyLSTMRegressor",
]
