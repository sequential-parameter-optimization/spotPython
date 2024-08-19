import lightning as L
from torch.utils.data import DataLoader, Subset
from typing import Optional
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, TensorDataset
import torch


class LightCrossValidationDataModule(L.LightningDataModule):
    """
    A LightningDataModule for handling cross-validation data splits.

    Args:
        batch_size (int): The size of the batch. Defaults to 64.
        k (int): The fold number. Defaults to 1.
        split_seed (int): The random seed for splitting the data. Defaults to 42.
        num_splits (int): The number of splits for cross-validation. Defaults to 10.
        data_dir (str): The path to the dataset. Defaults to "./data".
        num_workers (int): The number of workers for data loading. Defaults to 0.
        pin_memory (bool): Whether to pin memory for data loading. Defaults to False.

    Attributes:
        data_train (Optional[Dataset]): The training dataset.
        data_val (Optional[Dataset]): The validation dataset.

    Examples:
        >>> from spotpython.light import LightCrossValidationDataModule
        >>> data_module = LightCrossValidationDataModule()
        >>> data_module.setup()
        >>> print(f"Training set size: {len(data_module.data_train)}")
        Training set size: 45000
        >>> print(f"Validation set size: {len(data_module.data_val)}")
        Validation set size: 5000
        >>> print(f"Test set size: {len(data_module.data_test)}")
        Test set size: 10000
    """

    def __init__(
        self,
        batch_size=64,
        dataset=None,
        k: int = 1,
        split_seed: int = 42,
        num_splits: int = 10,
        data_dir: str = "./data",
        num_workers: int = 0,
        pin_memory: bool = False,
        scaler: Optional[object] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_full = dataset
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.k = k
        self.split_seed = split_seed
        self.num_splits = num_splits
        self.pin_memory = pin_memory
        self.scaler = scaler
        self.save_hyperparameters(logger=False)
        assert 0 <= self.k < self.num_splits, "incorrect fold number"

        # no data transformations
        self.transforms = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Prepares the data for use."""
        # download
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the data for use.

        Args:
            stage (Optional[str]): The current stage. Defaults to None.
        """
        if not self.data_train and not self.data_val:
            dataset_full = self.data_full
            kf = KFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.split_seed)
            all_splits = [k for k in kf.split(dataset_full)]
            train_indexes, val_indexes = all_splits[self.hparams.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
            self.data_train = Subset(dataset_full, train_indexes)
            print(f"Train Dataset Size: {len(self.data_train)}")
            self.data_val = Subset(dataset_full, val_indexes)
            print(f"Val Dataset Size: {len(self.data_val)}")

        if self.scaler is not None:
            # Fit the scaler on training data and transform both train and val data
            scaler_train_data = torch.stack([self.data_train[i][0] for i in range(len(self.data_train))]).squeeze(1)
            self.scaler.fit(scaler_train_data)
            self.data_train = [(self.scaler.transform(data), target) for data, target in self.data_train]
            data_tensors_train = [data.clone().detach() for data, target in self.data_train]
            target_tensors_train = [target.clone().detach() for data, target in self.data_train]
            self.data_train = TensorDataset(
                torch.stack(data_tensors_train).squeeze(1), torch.stack(target_tensors_train)
            )
            self.data_val = [(self.scaler.transform(data), target) for data, target in self.data_val]
            data_tensors_val = [data.clone().detach() for data, target in self.data_val]
            target_tensors_val = [target.clone().detach() for data, target in self.data_val]
            self.data_val = TensorDataset(torch.stack(data_tensors_val).squeeze(1), torch.stack(target_tensors_val))

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader.

        Returns:
            DataLoader: The training dataloader.

        Examples:
            >>> from spotpython.light import LightCrossValidationDataModule
            >>> data_module = LightCrossValidationDataModule()
            >>> data_module.setup()
            >>> train_dataloader = data_module.train_dataloader()
            >>> print(f"Training set size: {len(train_dataloader.dataset)}")
            Training set size: 45000

        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader.

        Returns:
            DataLoader: The validation dataloader.

        Examples:
            >>> from spotpython.light import LightCrossValidationDataModule
            >>> data_module = LightCrossValidationDataModule()
            >>> data_module.setup()
            >>> val_dataloader = data_module.val_dataloader()
            >>> print(f"Validation set size: {len(val_dataloader.dataset)}")
            Validation set size: 5000
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
