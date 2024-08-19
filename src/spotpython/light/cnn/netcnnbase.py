import lightning as L
import torch
from torch import nn
import torch.optim as optim
from spotpython.light.cnn.googlenet import GoogleNet


class NetCNNBase(L.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Initializes the CNN model.

        Args:
            model_name (str): name of the model.
            model_hparams (dict): dictionary containing the hyperparameters for the model.
            optimizer_name (str): name of the optimizer.
            optimizer_hparams (dict): dictionary containing the hyperparameters for the optimizer.

        Returns:
            (object): model object.

        Examples:
            >>> from spotpython.light.cnn.netcnnbase import NetCNNBase
                from spotpython.light.cnn.googlenet import GoogleNet
                import torch
                import torch.nn as nn
                model_hparams = {"c_in": 3, "c_out": 10, "act_fn": nn.ReLU, "optimizer_name": "Adam"}
                fun_control = {"core_model": GoogleNet}
                model = NetCNNBase(model_hparams, fun_control)
                x = torch.randn(1, 3, 32, 32)
                y = model(x)
                y.shape
                torch.Size([1, 10])

        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        print(f"model_hparams: {model_hparams}")
        print(f"self.hparams: {self.hparams}")
        # Create model
        self.model = self.create_model(model_name, model_hparams)
        # self.model = fun_control["core_model"](**model_hparams)
        print(f"self.model: {self.model}")
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
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

    def create_model(self, model_name, model_hparams):
        print("create_model: Starting")
        print(f"model_name: {model_name}")
        print(f"model_hparams: {model_hparams}")
        model_dict = {"GoogleNet": GoogleNet}
        if model_name in model_dict:
            return model_dict[model_name](**model_hparams)
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'
