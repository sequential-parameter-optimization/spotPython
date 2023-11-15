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
    test_result = trainer.test(datamodule=dm, ckpt_path="last")
    test_result = test_result[0]
    print(f"test_model result: {test_result}")
    return test_result["val_loss"], test_result["hp_metric"]
