import lightning as L
from spotpython.data.lightdatamodule import LightDataModule
from spotpython.utils.eda import generate_config_id
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
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
        >>> from spotpython.utils.init import fun_control_init
             from spotpython.light.netlightregression import NetLightRegression
            from spotpython.hyperdict.light_hyper_dict import LightHyperDict
            from spotpython.hyperparameters.values import (add_core_model_to_fun_control,
              get_default_hyperparameters_as_array)
            from spotpython.data.diabetes import Diabetes
            from spotpython.hyperparameters.values import set_control_key_value
            from spotpython.hyperparameters.values import (get_var_name, assign_values,
                generate_one_config_from_var_dict)
            import spotpython.light.testmodel as tm
            fun_control = fun_control_init(
                _L_in=10,
                _L_out=1,
                _torchmetric="mean_squared_error")
            dataset = Diabetes()
            set_control_key_value(control_dict=fun_control,
                                    key="data_set",
                                    value=dataset)
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
    _L_cond = fun_control["_L_cond"]
    _torchmetric = fun_control["_torchmetric"]
    if fun_control["enable_progress_bar"] is None:
        enable_progress_bar = False
    else:
        enable_progress_bar = fun_control["enable_progress_bar"]
    # Add "TEST" postfix to config_id
    # config id is unique. Since the model is loaded from a checkpoint,
    # the config id is generated here without a timestamp. This differs from
    # the config id generated in cvmodel.py and trainmodel.py.
    config_id = generate_config_id(config, timestamp=False) + "_TEST"
    if fun_control["data_module"] is None:
        dm = LightDataModule(
            dataset=fun_control["data_set"],
            data_full_train=fun_control["data_full_train"],
            data_test=fun_control["data_test"],
            data_val=fun_control["data_val"],
            batch_size=config["batch_size"],
            num_workers=fun_control["num_workers"],
            test_size=fun_control["test_size"],
            test_seed=fun_control["test_seed"],
            scaler=fun_control["scaler"],
            collate_fn_name=fun_control["collate_fn_name"],
            shuffle_train=fun_control["shuffle_train"],
            shuffle_val=fun_control["shuffle_val"],
            shuffle_test=fun_control["shuffle_test"],
            verbosity=fun_control["verbosity"],
        )
    else:
        dm = fun_control["data_module"]
    # TODO: Check if this is necessary:
    # dm.setup()
    # Init model from datamodule's attributes
    model = fun_control["core_model"](**config, _L_in=_L_in, _L_out=_L_out, _L_cond=_L_cond, _torchmetric=_torchmetric)

    trainer = L.Trainer(
        # Where to save models
        default_root_dir=os.path.join(fun_control["CHECKPOINT_PATH"], config_id),
        max_epochs=model.hparams.epochs,
        accelerator=fun_control["accelerator"],
        devices=fun_control["devices"],
        strategy=fun_control["strategy"],
        num_nodes=fun_control["num_nodes"],
        precision=fun_control["precision"],
        logger=TensorBoardLogger(
            save_dir=fun_control["TENSORBOARD_PATH"],
            version=config_id,
            default_hp_metric=True,
            log_graph=fun_control["log_graph"],
        ),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config["patience"], mode="min", strict=False, verbose=False),
            ModelCheckpoint(dirpath=os.path.join(fun_control["CHECKPOINT_PATH"], config_id), save_last=True),  # Save the last checkpoint
        ],
        enable_progress_bar=enable_progress_bar,
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model=model, datamodule=dm)

    # Load the last checkpoint
    # test_result = trainer.test(datamodule=dm, ckpt_path="last")
    test_result = trainer.test(datamodule=dm, ckpt_path="best")
    test_result = test_result[0]
    print(f"test_model result: {test_result}")
    return test_result["val_loss"], test_result["hp_metric"]
