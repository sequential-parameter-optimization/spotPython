import lightning as L
from spotPython.light.mnistdatamodule import MNISTDataModule
from spotPython.light.litmodel import LitModel


def train_model(config, fun_control):
    # print("Starting train_model")
    # model = fun_control["core_model"](
    #     **config, train=fun_control["train"], test=fun_control["test"], target_column=fun_control["target_column"]
    # )
    # print(f"model: {model}")

    # logger = TensorBoardLogger(save_dir="./runs/28", version=1, name="lightning_logs")
    # trainer = L.Trainer(
    #     accelerator="auto",
    #     devices=1,
    #     max_epochs=model.epochs,
    #     logger=logger,
    #     # logger=CSVLogger(save_dir="logs/"),
    # )
    # print(f"trainer initialized: {trainer}")
    # trainer.fit(model)
    # print("trainer.fit(model) completed")
    # result = trainer.test(model)
    # print(f"train_model result: {result}")
    # return result

    # Init DataModule
    dm = MNISTDataModule()
    # Init model from datamodule's attributes
    model = LitModel(*dm.dims, dm.num_classes)
    # Init trainer
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model=model, datamodule=dm)
    # Test best model on validation and test set
    result = trainer.validate(model=model, datamodule=dm, ckpt_path="last")
    # unlist the result (from a list of one dict)
    result = result[0]
    print(f"train_model result: {result}")
    return result["val_loss"]
