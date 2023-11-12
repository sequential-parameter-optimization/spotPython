import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from typing import Optional


class LightDataModule(L.LightningDataModule):
    """
    A LightningDataModule for handling data.

    Args:
        batch_size (int): The batch size.
        dataset (Dataset): The dataset.
        test_size (float): The test size. Defaults to 0.6.
        test_seed (int): The test seed. Defaults to 42.
        num_workers (int): The number of workers. Defaults to 0.

    Attributes:
        batch_size (int): The batch size.
        data_full (Dataset): The full dataset.
        data_test (Dataset): The test dataset.
        data_train (Dataset): The training dataset.
        data_val (Dataset): The validation dataset.
        num_workers (int): The number of workers.
        test_seed (int): The test seed.
        test_size (float): The test size.

    Examples:
        >>> from spotPython.data.lightdatamodule import LightDataModule
            from spotPython.data.csvdataset import CSVDataset
            from spotPython.data.pkldataset import PKLDataset
            import torch
            dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
            data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
            data_module.setup()
            print(f"Training set size: {len(data_module.data_train)}")
            Training set size: 3

    """

    def __init__(
        self, batch_size: int, dataset=None, test_size: float = 0.6, test_seed: int = 42, num_workers: int = 0
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_full = dataset
        self.test_size = test_size
        self.test_seed = test_seed
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
        >>> from spotPython.data.lightdatamodule import LightDataModule
            from spotPython.data.csvdataset import CSVDataset
            from spotPython.data.pkldataset import PKLDataset
            import torch
            dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
            data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
            data_module.setup()
            print(f"Training set size: {len(data_module.data_train)}")
            Training set size: 3

        """
        # if test_size is float, then train_size is 1 - test_size
        test_size = self.test_size
        if isinstance(self.test_size, float):
            full_train_size = round(1.0 - test_size, 2)
            val_size = round(full_train_size * test_size, 2)
            train_size = round(full_train_size - val_size, 2)
        else:
            # if test_size is int, then train_size is len(data_full) - test_size
            full_train_size = len(self.data_full) - test_size
            val_size = int(full_train_size * test_size / len(self.data_full))
            train_size = full_train_size - val_size

        print(f"full_train_size: {full_train_size}")
        print(f"val_size: {val_size}")
        print(f"train_size: {train_size}")
        print(f"test_size: {test_size}")

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.data_train, self.data_val, _ = random_split(self.data_full, [train_size, val_size, test_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            # get test data aset as test_abs percent of the full dataset
            generator_test = torch.Generator().manual_seed(self.test_seed)
            self.data_test, _ = random_split(self.data_full, [test_size, full_train_size], generator=generator_test)

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader.

        Returns:
            DataLoader: The training dataloader.

        Examples:
        >>> from spotPython.data.lightdatamodule import LightDataModule
            from spotPython.data.csvdataset import CSVDataset
            from spotPython.data.pkldataset import PKLDataset
            import torch
            dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
            data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
            data_module.setup()
            print(f"Training set size: {len(data_module.data_train)}")
            Training set size: 3

        """
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader.

        Returns:
            DataLoader: The validation dataloader.

        Examples:
        >>> from spotPython.data.lightdatamodule import LightDataModule
            from spotPython.data.csvdataset import CSVDataset
            from spotPython.data.pkldataset import PKLDataset
            import torch
            dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
            data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
            data_module.setup()
            print(f"Training set size: {len(data_module.data_val)}")
            Training set size: 3

        """
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.

        Returns:
            DataLoader: The test dataloader.

        Examples:
        >>> from spotPython.data.lightdatamodule import LightDataModule
            from spotPython.data.csvdataset import CSVDataset
            from spotPython.data.pkldataset import PKLDataset
            import torch
            dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
            data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
            data_module.setup()
            print(f"Test set size: {len(data_module.data_test)}")
            Test set size: 6

        """
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)
