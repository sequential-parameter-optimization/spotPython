import lightning as L
from torch.utils.data import DataLoader, random_split
from spotPython.light.csvdataset import CSVDataset
from typing import Optional


class CSVDataModule(L.LightningDataModule):
    """
    A LightningDataModule for handling CSV data.

    Args:
        batch_size (int): The size of the batch.
        DATASET_PATH (str): The path to the dataset. Defaults to "./data".
        num_workers (int): The number of workers for data loading. Defaults to 0.

    Attributes:
        data_train (Dataset): The training dataset.
        data_val (Dataset): The validation dataset.
        data_test (Dataset): The test dataset.
    """

    def __init__(self, batch_size: int, DATASET_PATH: str = "./data", num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """Prepares the data for use."""
        # download
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the data for use.

        Args:
            stage (Optional[str]): The current stage. Defaults to None.
        Examples:
            >>> from spotPython.light import CSVDataModule
            >>> data_module = CSVDataModule(batch_size=64)
            >>> data_module.setup()
            >>> print(f"Training set size: {len(data_module.data_train)}")
            Training set size: 45000
            >>> print(f"Validation set size: {len(data_module.data_val)}")
            Validation set size: 5000
            >>> print(f"Test set size: {len(data_module.data_test)}")
            Test set size: 10000

        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            data_full = CSVDataset(csv_file="./data/VBDP/train.csv", train=True)
            test_abs = int(len(data_full) * 0.6)
            self.data_train, self.data_val = random_split(data_full, [test_abs, len(data_full) - test_abs])

        # Assign test dataset for use in dataloader(s)
        # TODO: Adapt this to the VBDP Situation
        if stage == "test" or stage is None:
            self.data_test = CSVDataset(csv_file="./data/VBDP/train.csv", train=True)

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader.

        Returns:
            DataLoader: The training dataloader.
        Examples:
            >>> from spotPython.light import CSVDataModule
            >>> data_module = CSVDataModule(batch_size=64)
            >>> data_module.setup()
            >>> train_dataloader = data_module.train_dataloader()
            >>> print(f"Training dataloader size: {len(train_dataloader)}")
            Training dataloader size: 704

        """
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader.

        Returns:
            DataLoader: The validation dataloader.
        Examples:
            >>> from spotPython.light import CSVDataModule
            >>> data_module = CSVDataModule(batch_size=64)
            >>> data_module.setup()
            >>> val_dataloader = data_module.val_dataloader()
            >>> print(f"Validation dataloader size: {len(val_dataloader)}")
            Validation dataloader size: 79

        """
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.

        Returns:
            DataLoader: The test dataloader.

        Examples:
            >>> from spotPython.light import CSVDataModule
            >>> data_module = CSVDataModule(batch_size=64)
            >>> data_module.setup()
            >>> test_dataloader = data_module.test_dataloader()
            >>> print(f"Test dataloader size: {len(test_dataloader)}")
            Test dataloader size: 704

        """
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)
