import lightning as L
from spotPython.data.lightcrossvalidationdatamodule import LightCrossValidationDataModule
from spotPython.utils.eda import generate_config_id
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from spotPython.torch.initialization import kaiming_init, xavier_init
import os


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
    config_id = generate_config_id(config, timestamp=True) + "_CV"
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

        dm = LightCrossValidationDataModule(
            k=k,
            num_splits=num_folds,
            split_seed=split_seed,
            dataset=fun_control["data_set"],
            num_workers=fun_control["num_workers"],
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
            accelerator=fun_control["accelerator"],
            devices=fun_control["devices"],
            logger=TensorBoardLogger(
                save_dir=fun_control["TENSORBOARD_PATH"],
                version=config_id,
                default_hp_metric=True,
                log_graph=fun_control["log_graph"],
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

        results.append(score["val_loss"])

    score = sum(results) / num_folds
    # print(f"cv_model mapk result: {mapk_score}")
    return score
