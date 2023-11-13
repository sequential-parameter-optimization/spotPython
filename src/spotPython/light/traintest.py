import lightning as L

# from spotPython.light.csvdatamodule import CSVDataModule
from spotPython.data.lightdatamodule import LightDataModule
from spotPython.light.crossvalidationdatamodule import CrossValidationDataModule
from spotPython.utils.eda import generate_config_id
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from spotPython.torch.initialization import kaiming_init, xavier_init
import os
from typing import Tuple, Any


def train_model(config: dict, fun_control: dict) -> float:
    """
    Trains a model using the given configuration and function control parameters.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model.
        fun_control (dict): A dictionary containing the function control parameters.

    Returns:
        float: The validation loss of the trained model.

    Examples:
        >>> config = {
        ...     "initialization": "Xavier",
        ...     "batch_size": 32,
        ...     "patience": 10,
        ... }
        >>> fun_control = {
        ...     "_L_in": 10,
        ...     "_L_out": 1,
        ...     "enable_progress_bar": True,
        ...     "core_model": MyModel,
        ...     "num_workers": 4,
        ...     "DATASET_PATH": "./data",
        ...     "CHECKPOINT_PATH": "./checkpoints",
        ...     "TENSORBOARD_PATH": "./tensorboard",
        ... }
        >>> val_loss = train_model(config, fun_control)
    """
    _L_in = fun_control["_L_in"]
    _L_out = fun_control["_L_out"]
    # print(f"_L_in: {_L_in}")
    # print(f"_L_out: {_L_out}")
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
    # print(f"model: {model}")

    # # Init DataModule
    # dm = CSVDataModule(
    #     batch_size=config["batch_size"],
    #     num_workers=fun_control["num_workers"],
    #     data_dir=fun_control["DATASET_PATH"],
    # )

    print(fun_control["data_set"].data.shape)
    print(fun_control["data_set"].targets.shape)
    dm = LightDataModule(
        dataset=fun_control["data_set"],
        batch_size=config["batch_size"],
        num_workers=fun_control["num_workers"],
    )
    dm.setup()
    print(f"train_model(): Test set size: {len(dm.data_test)}")

    # Init trainer
    trainer = L.Trainer(
        # Where to save models
        default_root_dir=os.path.join(fun_control["CHECKPOINT_PATH"], config_id),
        max_epochs=model.hparams.epochs,
        accelerator="auto",
        devices=1,
        logger=TensorBoardLogger(save_dir=fun_control["TENSORBOARD_PATH"], version=config_id, default_hp_metric=True),
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
    # print(f"train_model result: {result}")
    return result["val_loss"]


def test_model(config: dict, fun_control: dict) -> Tuple[float, float]:
    """
    Tests a model using the given configuration and function control parameters.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model.
        fun_control (dict): A dictionary containing the function control parameters.

    Returns:
        Tuple[float, float]: The validation loss and accuracy of the tested model.

    Examples:
        >>> config = {
        ...     "initialization": "Xavier",
        ...     "batch_size": 32,
        ...     "patience": 10,
        ... }
        >>> fun_control = {
        ...     "_L_in": 10,
        ...     "_L_out": 1,
        ...     "enable_progress_bar": True,
        ...     "core_model": MyModel,
        ...     "num_workers": 4,
        ...     "DATASET_PATH": "./data",
        ...     "CHECKPOINT_PATH": "./checkpoints",
        ...     "TENSORBOARD_PATH": "./tensorboard",
        ... }
        >>> val_loss, val_acc = test_model(config, fun_control)
    """
    _L_in = fun_control["_L_in"]
    _L_out = fun_control["_L_out"]
    if fun_control["enable_progress_bar"] is None:
        enable_progress_bar = False
    else:
        enable_progress_bar = fun_control["enable_progress_bar"]
    # Add "TEST" postfix to config_id
    config_id = generate_config_id(config) + "_TEST"
    # Init DataModule
    # dm = CSVDataModule(
    #     batch_size=config["batch_size"],
    #     num_workers=fun_control["num_workers"],
    #     data_dir=fun_control["DATASET_PATH"],
    # )

    print("\n******\nIn test_model:", fun_control["data_set"].data.shape)
    print(fun_control["data_set"].targets.shape)
    dm = LightDataModule(
        dataset=fun_control["data_set"],
        batch_size=config["batch_size"],
        num_workers=fun_control["num_workers"],
    )
    dm.setup()
    print(f"Test set size: {len(dm.data_test)}")

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
        logger=TensorBoardLogger(save_dir=fun_control["TENSORBOARD_PATH"], version=config_id, default_hp_metric=True),
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
    # print(f"test_model result: {test_result}")
    return test_result["val_loss"], test_result["val_acc"]


def cv_model(config: dict, fun_control: dict) -> float:
    """
    Performs k-fold cross-validation on a model using the given configuration and function control parameters.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model.
        fun_control (dict): A dictionary containing the function control parameters.

    Returns:
        (float): The mean average precision at k (MAP@k) score of the model.

    Examples:
        >>> config = {
        ...     "initialization": "Xavier",
        ...     "batch_size": 32,
        ...     "patience": 10,
        ... }
        >>> fun_control = {
        ...     "_L_in": 10,
        ...     "_L_out": 1,
        ...     "enable_progress_bar": True,
        ...     "core_model": MyModel,
        ...     "num_workers": 4,
        ...     "DATASET_PATH": "./data",
        ...     "CHECKPOINT_PATH": "./checkpoints",
        ...     "TENSORBOARD_PATH": "./tensorboard",
        ...     "k_folds": 5,
        ... }
        >>> mapk_score = cv_model(config, fun_control)
    """
    _L_in = fun_control["_L_in"]
    _L_out = fun_control["_L_out"]
    if fun_control["enable_progress_bar"] is None:
        enable_progress_bar = False
    else:
        enable_progress_bar = fun_control["enable_progress_bar"]
    # Add "CV" postfix to config_id
    config_id = generate_config_id(config) + "_CV"
    results = []
    num_folds = fun_control["k_folds"]
    split_seed = 12345

    for k in range(num_folds):
        print("k:", k)

        model = fun_control["core_model"](**config, _L_in=_L_in, _L_out=_L_out)
        initialization = config["initialization"]
        if initialization == "Xavier":
            xavier_init(model)
        elif initialization == "Kaiming":
            kaiming_init(model)
        else:
            pass
        # print(f"model: {model}")

        dm = CrossValidationDataModule(
            k=k,
            num_splits=num_folds,
            split_seed=split_seed,
            batch_size=config["batch_size"],
            data_dir=fun_control["DATASET_PATH"],
        )
        dm.prepare_data()
        dm.setup()

        # Init trainer
        trainer = L.Trainer(
            # Where to save models
            default_root_dir=os.path.join(fun_control["CHECKPOINT_PATH"], config_id),
            max_epochs=model.hparams.epochs,
            accelerator="auto",
            devices=1,
            logger=TensorBoardLogger(
                save_dir=fun_control["TENSORBOARD_PATH"], version=config_id, default_hp_metric=True
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
        score = trainer.validate(model=model, datamodule=dm)
        # unlist the result (from a list of one dict)
        score = score[0]
        print(f"train_model result: {score}")

        results.append(score["valid_mapk"])

    mapk_score = sum(results) / num_folds
    # print(f"cv_model mapk result: {mapk_score}")
    return mapk_score


def load_light_from_checkpoint(config: dict, fun_control: dict, postfix: str = "_TEST") -> Any:
    """
    Loads a model from a checkpoint using the given configuration and function control parameters.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model.
        fun_control (dict): A dictionary containing the function control parameters.
        postfix (str): The postfix to append to the configuration ID when generating the checkpoint path.

    Returns:
        Any: The loaded model.

    Examples:
        >>> config = {
        ...     "initialization": "Xavier",
        ...     "batch_size": 32,
        ...     "patience": 10,
        ... }
        >>> fun_control = {
        ...     "_L_in": 10,
        ...     "_L_out": 1,
        ...     "core_model": MyModel,
        ...     "TENSORBOARD_PATH": "./tensorboard",
        ... }
        >>> model = load_light_from_checkpoint(config, fun_control)
    """
    config_id = generate_config_id(config) + postfix
    # default_root_dir = fun_control["TENSORBOARD_PATH"] + "lightning_logs/" + config_id + "/checkpoints/last.ckpt"
    default_root_dir = os.path.join(fun_control["CHECKPOINT_PATH"], config_id, "last.ckpt")
    # default_root_dir = os.path.join(fun_control["CHECKPOINT_PATH"], config_id)
    print(f"Loading model from {default_root_dir}")
    model = fun_control["core_model"].load_from_checkpoint(
        default_root_dir, _L_in=fun_control["_L_in"], _L_out=fun_control["_L_out"]
    )
    # disable randomness, dropout, etc...
    model.eval()
    return model
