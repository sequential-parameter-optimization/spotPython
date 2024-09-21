import lightning as L
from spotpython.data.lightdatamodule import LightDataModule
from spotpython.utils.eda import generate_config_id
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import os


def train_model(config: dict, fun_control: dict, timestamp: bool = True) -> float:
    """
    Trains a model using the given configuration and function control parameters.

    Args:
        config (dict):
            A dictionary containing the configuration parameters for the model.
        fun_control (dict):
            A dictionary containing the function control parameters.
        timestamp (bool):
            A boolean value indicating whether to include a timestamp in the config id. Default is True.
            If False, the string "_TRAIN" is appended to the config id.

    Returns:
        float: The validation loss of the trained model.

    Examples:
        >>> from spotpython.utils.init import fun_control_init
            from spotpython.light.netlightregression import NetLightRegression
            from spotpython.hyperdict.light_hyper_dict import LightHyperDict
            from spotpython.hyperparameters.values import (
                add_core_model_to_fun_control,
                get_default_hyperparameters_as_array)
            from spotpython.data.diabetes import Diabetes
            from spotpython.hyperparameters.values import set_control_key_value
            from spotpython.hyperparameters.values import get_var_name, assign_values, generate_one_config_from_var_dict
            from spotpython.light.traintest import train_model
            fun_control = fun_control_init(
                _L_in=10,
                _L_out=1,)
            # Select a dataset
            dataset = Diabetes()
            set_control_key_value(control_dict=fun_control,
                                key="data_set",
                                value=dataset)
            # Select a model
            add_core_model_to_fun_control(core_model=NetLightRegression,
                                        fun_control=fun_control,
                                        hyper_dict=LightHyperDict)
            # Select hyperparameters
            X = get_default_hyperparameters_as_array(fun_control)
            var_dict = assign_values(X, get_var_name(fun_control))
            for config in generate_one_config_from_var_dict(var_dict, fun_control):
                y = train_model(config, fun_control)
                break
            | Name   | Type       | Params | In sizes | Out sizes
            -------------------------------------------------------------
            0 | layers | Sequential | 157    | [16, 10] | [16, 1]
            -------------------------------------------------------------
            157       Trainable params
            0         Non-trainable params
            157       Total params
            0.001     Total estimated model params size (MB)
            Train_model(): Test set size: 266
            ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                Validate metric           DataLoader 0
            ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                    hp_metric             27462.841796875
                    val_loss              27462.841796875
            ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
            train_model result: {'val_loss': 27462.841796875, 'hp_metric': 27462.841796875}

    """
    _L_in = fun_control["_L_in"]
    _L_out = fun_control["_L_out"]
    _torchmetric = fun_control["_torchmetric"]
    if fun_control["enable_progress_bar"] is None:
        enable_progress_bar = False
    else:
        enable_progress_bar = fun_control["enable_progress_bar"]
    if timestamp:
        # config id is unique. Since the model is not loaded from a checkpoint,
        # the config id is generated here with a timestamp.
        config_id = generate_config_id(config, timestamp=True)
    else:
        # config id is not time-dependent and therefore unique,
        # so that the model can be loaded from a checkpoint,
        # the config id is generated here without a timestamp.
        config_id = generate_config_id(config, timestamp=False) + "_TRAIN"
    model = fun_control["core_model"](**config, _L_in=_L_in, _L_out=_L_out, _torchmetric=_torchmetric)

    dm = LightDataModule(
        dataset=fun_control["data_set"],
        batch_size=config["batch_size"],
        num_workers=fun_control["num_workers"],
        test_size=fun_control["test_size"],
        test_seed=fun_control["test_seed"],
        scaler=fun_control["scaler"],
    )
    # TODO: Check if this is necessary:
    # dm.setup()
    # print(f"train_model(): Test set size: {len(dm.data_test)}")
    # print(f"train_model(): Train set size: {len(dm.data_train)}")
    # print(f"train_model(): Batch size: {config['batch_size']}")

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config["patience"], mode="min", strict=False, verbose=False)
    ]
    if not timestamp:
        # add ModelCheckpoint only if timestamp is False
        callbacks.append(
            ModelCheckpoint(dirpath=os.path.join(fun_control["CHECKPOINT_PATH"], config_id), save_last=True)
        )  # Save the last checkpoint

    # Init trainer
    trainer = L.Trainer(
        # Where to save models
        default_root_dir=os.path.join(fun_control["CHECKPOINT_PATH"], config_id),
        max_epochs=model.hparams.epochs,
        accelerator=fun_control["accelerator"],
        devices=fun_control["devices"],
        logger=TensorBoardLogger(
            save_dir=fun_control["TENSORBOARD_PATH"],
            version=config_id,
            default_hp_metric=True,
            log_graph=fun_control["log_graph"],
        ),
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model=model, datamodule=dm)
    # Test best model on validation and test set
    # result = trainer.validate(model=model, datamodule=dm, ckpt_path="last")
    verbose = fun_control["verbosity"] > 0
    result = trainer.validate(model=model, datamodule=dm, verbose=verbose)
    # unlist the result (from a list of one dict)
    result = result[0]
    print(f"train_model result: {result}")
    return result["val_loss"]
