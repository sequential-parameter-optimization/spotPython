import lightning as L
import os
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from spotPython.light.cnn.netcnnbase import NetCNNBase

# for data loading
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.utils.data as data
import torch


def train_model(config: dict, fun_control: dict):
    """
    Trains a model for the given configuration and control parameters.

    Args:
        config (dict):
            dictionary containing the configuration for the hyperparameter tuning.
        fun_control (dict):
            dictionary containing control parameters for the hyperparameter tuning.

    Returns:
        (object):
            model object.
        (dict):
            dictionary containing the evaluation results.

    Examples:
        >>> from spotPython.light.trainmodel import train_model
            config = {"c_in": 3,
                        "c_out": 10,
                        "act_fn": nn.ReLU,
                        "optimizer_name": "Adam",
                        "optimizer_hparams": {"lr": 1e-3, "weight_decay": 1e-4}}
            fun_control = {"core_model": GoogleNet}
            model, result = train_model(config, fun_control)
            result
            {'test': 0.8772, 'val': 0.8772}

    """
    print("train_model: Starting")
    print(f"train_model: config: {config}")
    save_name = "saved_models"
    # Create PyTorch Lightning data loaders
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/ConvNets")
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
    os.makedirs(DATASET_PATH, exist_ok=True)

    # Create PyTorch Lightning data loaders
    # TODO: Replace this by data loaders external to train_model method:

    train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
    DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(DATA_MEANS, DATA_STD)])
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(DATA_MEANS, DATA_STD),
        ]
    )
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    L.seed_everything(42)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    L.seed_everything(42)
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

    # Loading the test set
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(
        train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=4
    )
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    # END TODO

    # Create a PyTorch Lightning trainer with the generation callback
    print("train_model: Creating trainer")
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models
        # We run on a single GPU (if possible)
        accelerator="auto",
        devices=1,
        # How many epochs to train for if no patience is set
        max_epochs=4,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_acc"
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
        ],  # Log learning rate every epoch
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    print("train_model: Created trainer")

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = NetCNNBase.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)  # To be reproducable
        print("train_model: Creating model")
        model = NetCNNBase(config=config, fun_control=fun_control)  # Create model
        trainer.fit(model, train_loader, val_loader)
        model = NetCNNBase.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result
