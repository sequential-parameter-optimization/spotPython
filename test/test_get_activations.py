from torch.utils.data import DataLoader
from spotpython.utils.init import fun_control_init
from spotpython.data.diabetes import Diabetes
from spotpython.light.regression.netlightregression import NetLightRegression
from spotpython.hyperdict.light_hyper_dict import LightHyperDict
from spotpython.hyperparameters.values import get_default_hyperparameters_as_array, get_one_config_from_X
from spotpython.hyperparameters.values import set_control_key_value
from spotpython.plot.xai import get_activations
from spotpython.hyperparameters.values import add_core_model_to_fun_control
import numpy as np


def test_get_activations():
    # Initialize the function control
    fun_control = fun_control_init(_L_in=10, _L_out=1, _torchmetric="mean_squared_error")

    # Set up the dataset
    dataset = Diabetes()
    set_control_key_value(control_dict=fun_control, key="data_set", value=dataset, replace=True)

    # Add the core model to the function control
    add_core_model_to_fun_control(fun_control=fun_control, core_model=NetLightRegression, hyper_dict=LightHyperDict)

    # Get the default hyperparameters as an array
    X = get_default_hyperparameters_as_array(fun_control)
    print(f"X = {X}")

    # Get one configuration from the hyperparameter array
    config = get_one_config_from_X(X, fun_control)
    print(f"config = {config}")

    # Set the input and output layer sizes
    _L_in = fun_control["_L_in"]
    _L_out = fun_control["_L_out"]
    _torchmetric = fun_control["_torchmetric"]

    # Create the model
    model = fun_control["core_model"](**config, _L_in=_L_in, _L_out=_L_out, _torchmetric=_torchmetric)
    print(f"model = {model}")

    # config.update({"batch_size": 1})
    # Set the batch size
    batch_size = config["batch_size"]
    print(f"batch_size = {batch_size}")

    # Create the data loader
    _ = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Get the activations
    activations = get_activations(model, fun_control=fun_control, batch_size=batch_size, device="cpu")

    # Assert that the activations dictionary is not empty
    assert len(activations) > 0

    # Assert that the activations are numpy arrays
    for layer_index, activation in activations.items():
        assert isinstance(activation, np.ndarray)

    # TODO: Assert that the activations have the correct shape
    for layer_index, activation in activations.items():
        # only test the first layer, because the other layers have a different shape
        # Layer sizes are divided by 2, 2, and 4, respectively.
        # This depends on the network layout and might change in the future.
        # However, l1 should remain the same, because it is the first layer.
        # Activations depend on the batch_size and the layer size.
        expected_shape = (batch_size * config["l1"],)
        break
    assert activation.shape == expected_shape


test_get_activations()
