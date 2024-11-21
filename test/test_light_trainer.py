import pytest
import lightning as L
import numpy as np
import torch
from spotpython.data.diabetes import Diabetes
from spotpython.hyperdict.light_hyper_dict import LightHyperDict
from spotpython.utils.init import fun_control_init
from spotpython.hyperparameters.values import assign_values, generate_one_config_from_var_dict, get_var_name
from spotpython.data.lightdatamodule import LightDataModule
from spotpython.utils.scaler import TorchStandardScaler

@pytest.fixture
def setup_trainer_and_lightdatamodule():
    PREFIX = "000"
    data_set = Diabetes()
    fun_control = fun_control_init(
        PREFIX=PREFIX,
        fun_evals=float('inf'),
        max_time=1,
        data_set=data_set,
        core_model_name="light.regression.NNLinearRegressor",
        hyperdict=LightHyperDict,
        _L_in=10,
        _L_out=1
    )
    
    # Prepare config
    X = np.array([[3.0e+00, 5.0, 4.0e+00, 2.0e+00, 1.1e+01, 1.0e-02, 1.0e+00, 1.0e+01, 0.0e+00, 0.0e+00]])
    var_dict = assign_values(X, get_var_name(fun_control))
    config = list(generate_one_config_from_var_dict(var_dict, fun_control))[0]
    
    # Setup model and datamodule
    _torchmetric = "mean_squared_error"
    model = fun_control["core_model"](**config, _L_in=10, _L_out=1, _L_cond=None, _torchmetric=_torchmetric)
    dm = LightDataModule(
        dataset=data_set,
        batch_size=16,
        test_size=0.6,
        scaler=TorchStandardScaler()
    )
    
    return model, dm

def test_trainer_fit_and_validate(setup_trainer_and_lightdatamodule):
    model, dm = setup_trainer_and_lightdatamodule
    
    # Initialize the trainer
    trainer = L.Trainer(
        max_epochs=2,  # Small number for testing purposes
        enable_progress_bar=False
    )
    
    # Run fit and validate
    trainer.fit(model=model, datamodule=dm, ckpt_path=None)
    results = trainer.validate(model=model, datamodule=dm, ckpt_path=None)
    
    # Assertions to check the validation results
    assert results is not None
    assert len(results) > 0  # Make sure it contains validation metrics
    assert 'val_loss' in results[0]  # Assuming 'val_loss' is logged
