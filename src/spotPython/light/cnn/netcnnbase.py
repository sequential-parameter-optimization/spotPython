import lightning as L
import torch
from torch import nn

# from spotPython.light.utils import create_model
import torch.optim as optim

# from spotPython.light.cnn.googlenet import GoogleNet
import spotPython.light.cnn.googlenet


class NetCNNBase(L.LightningModule):
    def __init__(self, config, fun_control):
        """
        Initializes the CNN model.

        Args:
            config (dict): dictionary containing the configuration for the hyperparameter tuning.
            fun_control (dict): dictionary containing control parameters for the hyperparameter tuning.

        Returns:
            (object): model object.

        Examples:
            >>> from spotPython.light.cnn.netcnnbase import NetCNNBase
                from spotPython.light.cnn.googlenet import GoogleNet
                import torch
                import torch.nn as nn
                config = {"c_in": 3, "c_out": 10, "act_fn": nn.ReLU, "optimizer_name": "Adam"}
                fun_control = {"core_model": GoogleNet}
                model = NetCNNBase(config, fun_control)
                x = torch.randn(1, 3, 32, 32)
                y = model(x)
                y.shape
                torch.Size([1, 10])

        """
        print("NetCNNBase: Starting")
        print(f"NetCNNBase: config: {config}")
        print(f"NetCNNBase: fun_control['core_model']: {fun_control['core_model']}")
        config = {
            "c_in": 3,
            "c_out": 10,
            "act_fn": nn.ReLU,
            "optimizer_name": "Adam",
            "optimizer_hparams": {"lr": 1e-3, "weight_decay": 1e-4},
        }
        print("fun_control['core_model']: ", fun_control["core_model"])
        print("fun_control['core_model'].type: ", fun_control["core_model"].type)
        # fun_control = {"core_model": GoogleNet}
        fun_control = {"core_model": spotPython.light.cnn.googlenet.GoogleNet}
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()  # "fun_control" is not a hyperparameter )
        print(f"config: {config}")
        # Create model
        print("Creating model")
        # self.model = create_model(config, fun_control)
        self.model = fun_control["core_model"](**config)
        print("Model created")
        print(f"self.model: {self.model}")
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.config["optimizer_name"] == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.config["optimizer_hparams"])
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)
