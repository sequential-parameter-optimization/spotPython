import lightning as L
from spotPython.light.csvdatamodule import CSVDataModule
from spotPython.light.crossvalidationdatamodule import CrossValidationDataModule
from spotPython.utils.eda import generate_config_id
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from spotPython.torch.initialization import kaiming_init, xavier_init
import os


def train_model(config, fun_control):
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

    # Init DataModule
    dm = CSVDataModule(
        batch_size=config["batch_size"],
        num_workers=fun_control["num_workers"],
        DATASET_PATH=fun_control["DATASET_PATH"],
    )

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


def test_model(config, fun_control):
    _L_in = fun_control["_L_in"]
    _L_out = fun_control["_L_out"]
    if fun_control["enable_progress_bar"] is None:
        enable_progress_bar = False
    else:
        enable_progress_bar = fun_control["enable_progress_bar"]
    # Add "TEST" postfix to config_id
    config_id = generate_config_id(config) + "_TEST"
    # Init DataModule
    dm = CSVDataModule(
        batch_size=config["batch_size"],
        num_workers=fun_control["num_workers"],
        DATASET_PATH=fun_control["DATASET_PATH"],
    )
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
            ModelCheckpoint(save_last=True),  # Save the last checkpoint
        ],
        enable_progress_bar=enable_progress_bar,
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model=model, datamodule=dm)
    test_result = trainer.test(datamodule=dm, ckpt_path="last")
    test_result = test_result[0]
    # print(f"test_model result: {test_result}")
    return test_result["val_loss"], test_result["val_acc"]


def cv_model(config, fun_control):
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
            DATASET_PATH=fun_control["DATASET_PATH"],
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


def load_light_from_checkpoint(config, fun_control, postfix="_TEST"):
    config_id = generate_config_id(config) + postfix
    default_root_dir = fun_control["TENSORBOARD_PATH"] + "lightning_logs/" + config_id + "/checkpoints/last.ckpt"
    # default_root_dir = os.path.join(fun_control["CHECKPOINT_PATH"], config_id)
    print(f"Loading model from {default_root_dir}")
    model = fun_control["core_model"].load_from_checkpoint(
        default_root_dir, _L_in=fun_control["_L_in"], _L_out=fun_control["_L_out"]
    )
    # disable randomness, dropout, etc...
    model.eval()
    return model
