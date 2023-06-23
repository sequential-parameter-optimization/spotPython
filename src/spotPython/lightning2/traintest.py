from lightning.pytorch.loggers import TensorBoardLogger
from spotPython.lightning2.litmnist import LitMNIST
import lightning as L


def train_model():
    model_MNIST = LitMNIST(act_fn=Tanh(), batch_size=BATCH_SIZE)
    logger = TensorBoardLogger(save_dir="./runs/28", version=1, name="lightning_logs")
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=3,
        logger=logger,
        # logger=CSVLogger(save_dir="logs/"),
    )
    trainer.fit(model_MNIST)
    trainer.test(model=model_MNIST)
