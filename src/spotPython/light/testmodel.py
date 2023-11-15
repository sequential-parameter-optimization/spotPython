import lightning as L
from spotPython.data.lightdatamodule import LightDataModule
from spotPython.utils.eda import generate_config_id
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from spotPython.torch.initialization import kaiming_init, xavier_init
import os
from typing import Tuple


def test_model(config: dict, fun_control: dict) -> Tuple[float, float]:
    """
    Tests a model using the given configuration and function control parameters.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model.
        fun_control (dict): A dictionary containing the function control parameters.

    Returns:
        Tuple[float, float]: The validation loss and the hyperparameter metric of the tested model.

    Notes:
        * `test_model` saves the last checkpoint of the model from the training phase, which is called as follows:
            `trainer.fit(model=model, datamodule=dm)`.
        * The test result is evaluated with the following function call:
        `trainer.test(datamodule=dm, ckpt_path="last")`.

    Examples:
        >>> from spotPython.utils.init import fun_control_init
             from spotPython.light.netlightregression import NetLightRegression
            from spotPython.hyperdict.light_hyper_dict import LightHyperDict
            from spotPython.hyperparameters.values import (add_core_model_to_fun_control,
              get_default_hyperparameters_as_array)
            from spotPython.data.diabetes import Diabetes
            from spotPython.hyperparameters.values import set_data_set
            from spotPython.hyperparameters.values import (get_var_name, assign_values,
                generate_one_config_from_var_dict)
            import spotPython.light.testmodel as tm
            fun_control = fun_control_init(
                _L_in=10,
                _L_out=1,)
            dataset = Diabetes()
            set_data_set(fun_control=fun_control,
                            data_set=dataset)
            add_core_model_to_fun_control(core_model=NetLightRegression,
                                        fun_control=fun_control,
                                        hyper_dict=LightHyperDict)
            X = get_default_hyperparameters_as_array(fun_control)
            var_dict = assign_values(X, get_var_name(fun_control))
            for config in generate_one_config_from_var_dict(var_dict, fun_control):
                y_test = tm.test_model(config, fun_control)
    """
    _L_in = fun_control["_L_in"]
    _L_out = fun_control["_L_out"]
    if fun_control["enable_progress_bar"] is None:
        enable_progress_bar = False
    else:
        enable_progress_bar = fun_control["enable_progress_bar"]
    # Add "TEST" postfix to config_id
    config_id = generate_config_id(config) + "_TEST"
    dm = LightDataModule(
        dataset=fun_control["data_set"],
        batch_size=config["batch_size"],
        num_workers=fun_control["num_workers"],
    )
    dm.setup()
    # Init model from datamodule's attributes
    model = fun_control["core_model"](**config, _L_in=_L_in, _L_out=_L_out)
    initialization = config["initialization"]
    if initialization == "Xavier":
        xavier_init(model)
    elif initialization == "Kaiming":
        kaiming_init(model)
    else:
        pass
    # print(f"model: {model}")
    # Init trainer
    trainer = L.Trainer(
        # Where to save models
        default_root_dir=os.path.join(fun_control["CHECKPOINT_PATH"], config_id),
        max_epochs=model.hparams.epochs,
        accelerator="auto",
        devices=1,
        logger=TensorBoardLogger(
            save_dir=fun_control["TENSORBOARD_PATH"], version=config_id, default_hp_metric=True, log_graph=True
        ),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config["patience"], mode="min", strict=False, verbose=False),
            ModelCheckpoint(
                dirpath=os.path.join(fun_control["CHECKPOINT_PATH"], config_id), save_last=True
            ),  # Save the last checkpoint
        ],
        enable_progress_bar=enable_progress_bar,
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model=model, datamodule=dm)
    # Load the last checkpoint
    test_result = trainer.test(datamodule=dm, ckpt_path="last")
    test_result = test_result[0]
    print(f"test_model result: {test_result}")
    return test_result["val_loss"], test_result["hp_metric"]
