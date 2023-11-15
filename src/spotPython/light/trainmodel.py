import lightning as L
from spotPython.data.lightdatamodule import LightDataModule
from spotPython.utils.eda import generate_config_id
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from spotPython.torch.initialization import kaiming_init, xavier_init
import os


def train_model(config: dict, fun_control: dict) -> float:
    """
    Trains a model using the given configuration and function control parameters.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model.
        fun_control (dict): A dictionary containing the function control parameters.

    Returns:
        float: The validation loss of the trained model.

    Examples:
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.light.netlightregression import NetLightRegression
            from spotPython.hyperdict.light_hyper_dict import LightHyperDict
            from spotPython.hyperparameters.values import (
                add_core_model_to_fun_control,
                get_default_hyperparameters_as_array)
            from spotPython.data.diabetes import Diabetes
            from spotPython.hyperparameters.values import set_data_set
            from spotPython.hyperparameters.values import get_var_name, assign_values, generate_one_config_from_var_dict
            from spotPython.light.traintest import train_model
            fun_control = fun_control_init(
                _L_in=10,
                _L_out=1,)
            # Select a dataset
            dataset = Diabetes()
            set_data_set(fun_control=fun_control,
                            data_set=dataset)
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
    if fun_control["enable_progress_bar"] is None:
        enable_progress_bar = False
    else:
        enable_progress_bar = fun_control["enable_progress_bar"]
    config_id = generate_config_id(config)
    model = fun_control["core_model"](**config, _L_in=_L_in, _L_out=_L_out)
    initialization = config["initialization"]
    if initialization == "Xavier":
        xavier_init(model)
    elif initialization == "Kaiming":
        kaiming_init(model)
    else:
        pass

    dm = LightDataModule(
        dataset=fun_control["data_set"],
        batch_size=config["batch_size"],
        num_workers=fun_control["num_workers"],
    )
    dm.setup()
    print(f"Train_model(): Test set size: {len(dm.data_test)}")

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
            EarlyStopping(monitor="val_loss", patience=config["patience"], mode="min", strict=False, verbose=False)
        ],
        enable_progress_bar=enable_progress_bar,
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model=model, datamodule=dm)
    # Test best model on validation and test set
    # result = trainer.validate(model=model, datamodule=dm, ckpt_path="last")
    result = trainer.validate(model=model, datamodule=dm)
    # unlist the result (from a list of one dict)
    result = result[0]
    print(f"train_model result: {result}")
    return result["val_loss"]
