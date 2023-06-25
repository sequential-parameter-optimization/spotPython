import lightning as L
from spotPython.light.mnistdatamodule import MNISTDataModule
from spotPython.utils.eda import generate_config_id

# from spotPython.light.litmodel import LitModel
from pytorch_lightning.loggers import TensorBoardLogger


def train_model(config, fun_control):
    config_id = generate_config_id(config)
    # Init DataModule
    dm = MNISTDataModule(
        batch_size=config["batch_size"], num_workers=fun_control["num_workers"], data_dir=fun_control["data_dir"]
    )
    # Init model from datamodule's attributes
    # model = LitModel(*dm.dims, dm.num_classes)
    model = fun_control["core_model"](**config, _L_in=1 * 28 * 28, _L_out=10)
    print(f"model: {model}")
    # Init trainer
    trainer = L.Trainer(
        max_epochs=model.epochs,
        accelerator="auto",
        devices=1,
        logger=TensorBoardLogger(save_dir=fun_control["tensorboard_path"], version=config_id),
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
    config_id = generate_config_id(config)
    # Init DataModule
    dm = MNISTDataModule(
        batch_size=config["batch_size"], num_workers=fun_control["num_workers"], data_dir=fun_control["data_dir"]
    )
    # Init model from datamodule's attributes
    # model = LitModel(*dm.dims, dm.num_classes)
    model = fun_control["core_model"](**config, _L_in=1 * 28 * 28, _L_out=10)
    print(f"model: {model}")
    # Init trainer
    trainer = L.Trainer(
        max_epochs=model.epochs,
        accelerator="auto",
        devices=1,
        logger=TensorBoardLogger(save_dir=fun_control["tensorboard_path"], version=config_id),
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model=model, datamodule=dm)
    # Test best model on validation and test set
    # result = trainer.validate(model=model, datamodule=dm, ckpt_path="last")
    val_result = trainer.validate(model=model, datamodule=dm)
    # unlist the result (from a list of one dict)
    val_result = val_result[0]
    print(f"validation_model result: {val_result}")
    # test
    test_result = trainer.test(model=model, datamodule=dm)
    test_result = test_result[0]
    print(f"test_model result: {test_result}")
    return test_result["test_loss"]
