import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from typing import Optional


class LightDataModule(L.LightningDataModule):
    """
    A LightningDataModule for handling data.

    Args:
        batch_size (int):
            The batch size. Required.
        dataset (torch.utils.data.Dataset):
            The dataset from the torch.utils.data Dataset class.
            It  must implement three functions: __init__, __len__, and __getitem__.
            Required.
        test_size (float):
            The test size. if test_size is float, then train_size is 1 - test_size.
            If test_size is int, then train_size is len(data_full) - test_size.
            Train size will be split into train and validation sets.
            So if test size is 0.7, the 0.7 train size will be split into 0.7 * 0.7 = 0.49 train set
            amd 0.7 * 0.3 = 0.21 validation set.
        test_seed (int):
            The test seed. Defaults to 42.
        num_workers (int):
            The number of workers. Defaults to 0.

    Attributes:
        batch_size (int): The batch size.
        data_full (Dataset): The full dataset.
        data_test (Dataset): The test dataset.
        data_train (Dataset): The training dataset.
        data_val (Dataset): The validation dataset.
        num_workers (int): The number of workers.
        test_seed (int): The test seed.
        test_size (float): The test size.

    Methods:
        prepare_data(self):
            Usually used for downloading the data. Here: Does nothing, i.e., pass.
        setup(self, stage: Optional[str] = None):
            Performs the training, validation, and test split.
        train_dataloader():
            Returns a DataLoader instance for the training set.
        val_dataloader():
            Returns a DataLoader instance for the validation set.
        test_dataloader():
            Returns a DataLoader instance for the test set.

    Examples:
        >>> from spotPython.data.lightdatamodule import LightDataModule
            from spotPython.data.csvdataset import CSVDataset
            import torch
            # data.csv is simple csv file with 11 samples
            dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
            data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
            data_module.setup()
            print(f"Training set size: {len(data_module.data_train)}")
            print(f"Validation set size: {len(data_module.data_val)}")
            print(f"Test set size: {len(data_module.data_test)}")
            full_train_size: 0.5
            val_size: 0.25
            train_size: 0.25
            test_size: 0.5
            Training set size: 3
            Validation set size: 3
            Test set size: 6

    References:
        See https://lightning.ai/docs/pytorch/stable/data/datamodule.html

    """

    def __init__(self, batch_size: int, dataset: object, test_size: float, test_seed: int = 42, num_workers: int = 0):
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
        Splits the data for use in training, validation, and testing.
        Uses torch.utils.data.random_split() to split the data.
        Splitting is based on the test_size and test_seed.
        The test_size can be a float or an int.

        Args:
            stage (Optional[str]):
                The current stage. Can be "fit" (for training and validation), "test" (testing),
                or None (for all three stages). Defaults to None.

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

        print(f"LightDataModule: setup(). stage: {stage}")
        print(f"LightDataModule setup(): full_train_size: {full_train_size}")
        print(f"LightDataModule setup(): val_size: {val_size}")
        print(f"LightDataModule setup(): train_size: {train_size}")
        print(f"LightDataModule setup(): test_size: {test_size}")

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            print("LightDataModule: setup(). stage: fit")
            self.data_train, self.data_val, _ = random_split(self.data_full, [train_size, val_size, test_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            print("LightDataModule: setup(). stage: test")
            # get test data aset as test_abs percent of the full dataset
            generator_test = torch.Generator().manual_seed(self.test_seed)
            self.data_test, _ = random_split(self.data_full, [test_size, full_train_size], generator=generator_test)

        # if stage == "predict" or stage is None:
        #     print(f"test_size, full_train_size: {test_size}, {full_train_size}")
        #     generator_predict = torch.Generator().manual_seed(self.test_seed)
        #     full_data_predict, _ = random_split(
        #         self.data_full, [test_size, full_train_size], generator=generator_predict
        #     )
        #     # Only keep the features for prediction
        #     self.data_predict = [x for x, _ in full_data_predict]

        # Assign pred dataset for use in dataloader(s)
        if stage == "predict" or stage is None:
            print("LightDataModule: setup(). stage: predict")
            # get test data aset as test_abs percent of the full dataset
            generator_predict = torch.Generator().manual_seed(self.test_seed)
            self.data_predict, _ = random_split(
                self.data_full, [test_size, full_train_size], generator=generator_predict
            )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader, i.e., a pytorch DataLoader instance
        using the training dataset.

        Returns:
            DataLoader: The training dataloader.

        Examples:
            >>> from spotPython.data.lightdatamodule import LightDataModule
                from spotPython.data.csvdataset import CSVDataset
                import torch
                dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
                data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
                data_module.setup()
                print(f"Training set size: {len(data_module.data_train)}")
                Training set size: 3

        """
        print(f"LightDataModule: train_dataloader(). Training set size: {len(self.data_train)}")
        print(f"LightDataModule: train_dataloader(). batch_size: {self.batch_size}")
        print(f"LightDataModule: train_dataloader(). num_workers: {self.num_workers}")
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader, i.e., a pytorch DataLoader instance
        using the validation dataset.

        Returns:
            DataLoader: The validation dataloader.

        Examples:
            >>> from spotPython.data.lightdatamodule import LightDataModule
                from spotPython.data.csvdataset import CSVDataset
                import torch
                dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
                data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
                data_module.setup()
                print(f"Training set size: {len(data_module.data_val)}")
                Training set size: 3
        """
        print(f"LightDataModule: val_dataloader(). Training set size: {len(self.data_val)}")
        print(f"LightDataModule: val_dataloader(). batch_size: {self.batch_size}")
        print(f"LightDataModule: val_dataloader(). num_workers: {self.num_workers}")
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader, i.e., a pytorch DataLoader instance
        using the test dataset.

        Returns:
            DataLoader: The test dataloader.

        Examples:
            >>> from spotPython.data.lightdatamodule import LightDataModule
                from spotPython.data.csvdataset import CSVDataset
                import torch
                dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
                data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
                data_module.setup()
                print(f"Test set size: {len(data_module.data_test)}")
                Test set size: 6

        """
        print(f"LightDataModule: test_dataloader(). Test set size: {len(self.data_test)}")
        print(f"LightDataModule: test_dataloader(). batch_size: {self.batch_size}")
        print(f"LightDataModule: test_dataloader(). num_workers: {self.num_workers}")
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        """
        Returns the predict dataloader, i.e., a pytorch DataLoader instance
        using the predict dataset.

        Returns:
            DataLoader: The predict dataloader.

        Examples:
            >>> from spotPython.data.lightdatamodule import LightDataModule
                from spotPython.data.csvdataset import CSVDataset
                import torch
                dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
                data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
                data_module.setup()
                print(f"Predict set size: {len(data_module.data_predict)}")
                Predict set size: 6

        """
        print(f"LightDataModule: predict_dataloader(). Predict set size: {len(self.data_predict)}")
        print(f"LightDataModule: predict_dataloader(). batch_size: {self.batch_size}")
        print(f"LightDataModule: predict_dataloader(). num_workers: {self.num_workers}")
        return DataLoader(self.data_predict, batch_size=len(self.data_predict), num_workers=self.num_workers)
