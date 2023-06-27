import lightning as L
from spotPython.light.csvdatamodule import CSVDataModule
from spotPython.light.crossvalidationdatamodule import CrossValidationDataModule
from spotPython.utils.eda import generate_config_id
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from spotPython.torch.initialization import kaiming_init, xavier_init


def train_model(config, fun_control):
    if fun_control["enable_progress_bar"] is None:
        enable_progress_bar = False
    else:
        enable_progress_bar = fun_control["enable_progress_bar"]
    config_id = generate_config_id(config)
    # Init DataModule
    dm = CSVDataModule(
        batch_size=config["batch_size"], num_workers=fun_control["num_workers"], data_dir=fun_control["data_dir"]
    )
    # Init model from datamodule's attributes
    # model = LitModel(*dm.dims, dm.num_classes)
    model = fun_control["core_model"](**config, _L_in=64, _L_out=11)
    initialization = config["initialization"]
    if initialization == "Xavier":
        xavier_init(model)
    elif initialization == "Kaiming":
        kaiming_init(model)
    else:
        pass
    print(f"model: {model}")
    # Init trainer
    trainer = L.Trainer(
        max_epochs=model.epochs,
        accelerator="auto",
        devices=1,
        logger=TensorBoardLogger(save_dir=fun_control["tensorboard_path"], version=config_id, default_hp_metric=True),
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


def test_model(config, fun_control):
    if fun_control["enable_progress_bar"] is None:
        enable_progress_bar = False
    else:
        enable_progress_bar = fun_control["enable_progress_bar"]
    config_id = generate_config_id(config)
    # Init DataModule
    dm = CSVDataModule(
        batch_size=config["batch_size"], num_workers=fun_control["num_workers"], data_dir=fun_control["data_dir"]
    )
    # Init model from datamodule's attributes
    # model = LitModel(*dm.dims, dm.num_classes)
    model = fun_control["core_model"](**config, _L_in=64, _L_out=11)
    initialization = config["initialization"]
    if initialization == "Xavier":
        xavier_init(model)
    elif initialization == "Kaiming":
        kaiming_init(model)
    else:
        pass
    print(f"model: {model}")
    # Init trainer
    trainer = L.Trainer(
        max_epochs=model.epochs,
        accelerator="auto",
        devices=1,
        logger=TensorBoardLogger(save_dir=fun_control["tensorboard_path"], version=config_id, default_hp_metric=True),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config["patience"], mode="min", strict=False, verbose=False)
        ],
        enable_progress_bar=enable_progress_bar,
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model=model, datamodule=dm)
    test_result = trainer.test(datamodule=dm, ckpt_path="last")
    test_result = test_result[0]
    print(f"test_model result: {test_result}")
    return test_result["val_loss"], test_result["val_acc"]


def cv_model(config, fun_control):
    config_id = generate_config_id(config)
    if fun_control["enable_progress_bar"] is None:
        enable_progress_bar = False
    else:
        enable_progress_bar = fun_control["enable_progress_bar"]
    results = []
    num_folds = 10
    split_seed = 12345
    model = fun_control["core_model"](**config, _L_in=64, _L_out=11)
    initialization = config["initialization"]
    if initialization == "Xavier":
        xavier_init(model)
    elif initialization == "Kaiming":
        kaiming_init(model)
    else:
        pass
    print(f"model: {model}")

    for k in range(num_folds):
        print("k:", k)
        dm = CrossValidationDataModule(
            k=k,
            num_splits=num_folds,
            split_seed=split_seed,
            batch_size=config["batch_size"],
            data_dir=fun_control["data_dir"],
        )
        dm.prepare_data()
        dm.setup()

        # here we train the model on given split...
        print(f"model: {model}")
        # Init trainer
        trainer = L.Trainer(
            max_epochs=model.epochs,
            accelerator="auto",
            devices=1,
            logger=TensorBoardLogger(
                save_dir=fun_control["tensorboard_path"], version=config_id, default_hp_metric=True
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
    print(f"cv_model mapk result: {mapk_score}")
    return mapk_score
