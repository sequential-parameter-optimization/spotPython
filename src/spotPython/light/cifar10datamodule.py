import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from typing import Optional


class CIFAR10DataModule(pl.LightningDataModule):
    """
    A LightningDataModule for handling CIFAR10 data.

    Args:
        batch_size (int): The size of the batch.
        data_dir (str): The directory where the data is stored. Defaults to "./data".
        num_workers (int): The number of workers for data loading. Defaults to 0.

    Attributes:
        data_train (Dataset): The training dataset.
        data_val (Dataset): The validation dataset.
        data_test (Dataset): The test dataset.
    """

    def __init__(self, batch_size: int, data_dir: str = "./data", num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """Prepares the data for use."""
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the data for use.

        Args:
            stage (Optional[str]): The current stage. Defaults to None.

        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            transform = transforms.Compose([transforms.ToTensor()])
            cifar_full = CIFAR10(self.data_dir, train=True, transform=transform)
            self.data_train, self.data_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            transform = transforms.Compose([transforms.ToTensor()])
            self.data_test = CIFAR10(self.data_dir, train=False, transform=transform)

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader.

        Returns:
            DataLoader: The training dataloader.

        """
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader.

        Returns:
            DataLoader: The validation dataloader.


        """
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.

        Returns:
            DataLoader: The test dataloader.


        """
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)
