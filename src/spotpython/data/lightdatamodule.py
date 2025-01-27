import lightning as L
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from typing import Optional
from math import floor
from torch.nn.utils.rnn import pad_sequence


class PadSequenceManyToMany:
    """
    A callable class for padding sequences in a many-to-many RNN model.
    """

    def __call__(self, batch):
        batch_x, batch_y = zip(*batch)
        padded_batch_x = pad_sequence(batch_x, batch_first=True)
        padded_batch_y = pad_sequence(batch_y, batch_first=True)
        lengths = torch.tensor([len(x) for x in batch_x])

        return padded_batch_x, lengths, padded_batch_y


class PadSequenceManyToOne:
    """
    A callable class for padding sequences in a many-to-one RNN model.
    """

    def __call__(self, batch):
        batch_x, batch_y = zip(*batch)
        padded_batch_x = pad_sequence(batch_x, batch_first=True)
        lengths = torch.tensor([len(x) for x in batch_x])

        return padded_batch_x, lengths, torch.tensor(batch_y)


class LightDataModule(L.LightningDataModule):
    """
    A LightningDataModule for handling data.

    Args:
        batch_size (int):
            The batch size. Required.
        dataset (torch.utils.data.Dataset, optional):
            The dataset from the torch.utils.data Dataset class.
            It must implement three functions: __init__, __len__, and __getitem__.
        data_full_train (torch.utils.data.Dataset, optional):
            The full training dataset from which training and validation sets will be derived.
        data_test (torch.utils.data.Dataset, optional):
            The separate test dataset that will be used for testing.
        test_size (float, optional):
            The test size. If test_size is float, then train_size is 1 - test_size.
            If test_size is int, then train_size is len(data_full) - test_size.
        test_seed (int):
            The test seed. Defaults to 42.
        num_workers (int):
            The number of workers. Defaults to 0.
        scaler (object, optional):
            The spot scaler object (e.g. TorchStandardScaler). Defaults to None.
        verbosity (int):
            The verbosity level. Defaults to 0.

    Examples:
        >>> from spotpython.data.lightdatamodule import LightDataModule
            from spotpython.data.csvdataset import CSVDataset
            from spotpython.utils.scaler import TorchStandardScaler
            import torch
            # data.csv is simple csv file with 11 samples
            dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
            scaler = TorchStandardScaler()
            data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5, scaler=scaler)
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

    def __init__(
        self,
        batch_size: int,
        dataset: Optional[object] = None,
        data_full_train: Optional[object] = None,
        data_test: Optional[object] = None,
        data_val: Optional[object] = None,
        test_size: Optional[float] = None,
        test_seed: int = 42,
        collate_fn_name: Optional[str] = None,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        shuffle_test: bool = False,
        num_workers: int = 0,
        scaler: Optional[object] = None,
        verbosity: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_full = dataset
        self.data_full_train = data_full_train
        self.data_test = data_test
        self.data_val = data_val
        self.test_size = test_size
        self.test_seed = test_seed
        self.collate_fn_name = collate_fn_name
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test
        self.num_workers = num_workers
        self.scaler = scaler
        self.verbosity = verbosity

    def transform_dataset(self, dataset) -> TensorDataset:
        """Applies the scaler transformation to the dataset.

        Args:
            dataset (List[Tuple[torch.Tensor, Any]]): The dataset to transform, consisting of data and target pairs.

        Returns:
            TensorDataset: A PyTorch TensorDataset containing the transformed and cloned data and targets.

        Raises:
            ValueError: If the input data is not correctly formatted for transformation.
        """
        try:
            # Perform transformations on the data in a single iteration
            transformed_data = [(self.scaler.transform(data), target) for data, target in dataset]
            # Clone and detach data tensors
            data_tensors = [data.clone().detach() for data, _ in transformed_data]
            target_tensors = [target.clone().detach() for _, target in transformed_data]
            # Create a TensorDataset from the processed data
            return TensorDataset(torch.stack(data_tensors).squeeze(1), torch.stack(target_tensors))
        except Exception as e:
            raise ValueError(f"Error transforming dataset: {e}")

    def handle_scaling_and_transform(self) -> None:
        """
        Fits the scaler on the training data and transforms both training and validation datasets.
        This function is only called when self.scaler is not None.
        """
        # Ensure self.scaler is not None before proceeding
        if self.scaler is None:
            raise ValueError("Scaler object is required to perform scaling and transformation.")
        # Fit the scaler on training data
        scaler_train_data = torch.stack([self.data_train[i][0] for i in range(len(self.data_train))]).squeeze(1)
        if self.verbosity > 0:
            print(scaler_train_data.shape)
        self.scaler.fit(scaler_train_data)
        # Transform the training data
        self.data_train = self.transform_dataset(self.data_train)
        # Transform the validation data
        self.data_val = self.transform_dataset(self.data_val)

    def prepare_data(self) -> None:
        """Prepares the data for use."""
        # download
        pass

    def _setup_full_data_provided(self, stage) -> None:
        full_size = len(self.data_full)
        test_size = self.test_size

        # consider the case when test_size is a float
        if isinstance(self.test_size, float):
            full_train_size = 1.0 - self.test_size
            val_size = full_train_size * self.test_size
            train_size = full_train_size - val_size
        else:
            # test_size is an int, training size calculation directly based on it
            full_train_size = full_size - self.test_size
            val_size = floor(full_train_size * self.test_size / full_size)
            train_size = full_size - val_size - test_size

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            generator_fit = torch.Generator().manual_seed(self.test_seed)
            self.data_train, self.data_val, _ = random_split(self.data_full, [train_size, val_size, test_size], generator=generator_fit)
            if self.verbosity > 0:
                print(f"train_size: {train_size}, val_size: {val_size}, test_sie: {test_size} for splitting train & val data.")
                print(f"train samples: {len(self.data_train)}, val samples: {len(self.data_val)} generated for train & val data.")
            # Handle scaling and transformation if scaler is provided
            if self.scaler is not None:
                self.handle_scaling_and_transform()

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            generator_test = torch.Generator().manual_seed(self.test_seed)
            self.data_test, _, _ = random_split(self.data_full, [test_size, train_size, val_size], generator=generator_test)
            if self.verbosity > 0:
                print(f"train_size: {train_size}, val_size: {val_size}, test_sie: {test_size} for splitting test data.")
                print(f"test samples: {len(self.data_test)} generated for test data.")
            if self.scaler is not None:
                # Transform the test data
                self.data_test = self.transform_dataset(self.data_test)

        # Assign pred dataset for use in dataloader(s)
        if stage == "predict" or stage is None:
            generator_predict = torch.Generator().manual_seed(self.test_seed)
            self.data_predict, _, _ = random_split(self.data_full, [test_size, train_size, val_size], generator=generator_predict)
            if self.verbosity > 0:
                print(f"train_size: {train_size}, val_size: {val_size}, test_size (= predict_size): {test_size} for splitting predict data.")
                print(f"predict samples: {len(self.data_predict)} generated for predict data.")
            if self.scaler is not None:
                # Transform the predict data
                self.data_predict = self.transform_dataset(self.data_predict)

    def _setup_test_data_provided(self, stage) -> None:
        # New functionality with separate full_train and test datasets. Use these datasets directly.
        full_train_size = len(self.data_full_train)
        test_size = self.test_size
        # consider the case when test_size is a float
        if isinstance(self.test_size, float):
            val_size = self.test_size
            train_size = 1 - self.test_size
        else:
            # test_size is an int, training size calculation directly based on it
            full_size = len(self.data_full_train) + len(self.data_test)
            full_train_size = len(self.data_full_train)
            val_size = floor(full_train_size * self.test_size / full_size)
            train_size = full_train_size - val_size

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.verbosity > 0:
                print(f"train_size: {train_size}, val_size: {val_size} used for train & val data.")
            generator_fit = torch.Generator().manual_seed(self.test_seed)
            self.data_train, self.data_val = random_split(self.data_full_train, [train_size, val_size], generator=generator_fit)
            # Handle scaling and transformation if scaler is provided
            if self.scaler is not None:
                self.handle_scaling_and_transform()

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            if self.verbosity > 0:
                print(f"test_size: {test_size} used for test dataset.")
            self.data_test = self.data_test
            if self.scaler is not None:
                # Transform the test data
                self.data_test = self.transform_dataset(self.data_test)

        # Assign pred dataset for use in dataloader(s)
        if stage == "predict" or stage is None:
            if self.verbosity > 0:
                print(f"test_size: {test_size} used for predict dataset.")
            self.data_predict = self.data_test
            if self.scaler is not None:
                # Transform the predict data
                self.data_predict = self.transform_dataset(self.data_predict)

    def _setup_val_data_provided(self, stage) -> None:
        # New functionality for predefined train, validation and test data in the fun_control
        # Get the data set sizes
        if self.data_full_train is None:
            raise ValueError("If data_val is defined, data_full_train must also be defined.")
        train_size = len(self.data_full_train)
        val_size = len(self.data_val)
        test_size = len(self.data_test)
        # Assign train and validation data sets
        if stage == "fit" or stage is None:
            if self.verbosity > 0:
                print(f"train_size: {train_size}, val_size: {val_size} used for train & val data.")
            # Use all data from data_full_train as training data
            self.data_train = self.data_full_train
            # Handle scaling and transformation if scaler is provided
            if self.scaler is not None:
                self.handle_scaling_and_transform()

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            if self.verbosity > 0:
                print(f"test_size: {test_size} used for test dataset.")
            self.data_test = self.data_test
            if self.scaler is not None:
                # Transform the test data
                self.data_test = self.transform_dataset(self.data_test)

        # Assign pred dataset for use in dataloader(s)
        if stage == "predict" or stage is None:
            if self.verbosity > 0:
                print(f"test_size: {test_size} used for predict dataset.")
            self.data_predict = self.data_test
            if self.scaler is not None:
                # Transform the predict data
                self.data_predict = self.transform_dataset(self.data_predict)

    def _get_collate_fn(self, collate_fn_name: str = None) -> object:
        """
        Returns the collate function based on the collate_fn_name.
        """
        if collate_fn_name == "PadSequenceManyToMany":
            collate_fn = PadSequenceManyToMany()
        elif collate_fn_name == "PadSequenceManyToOne":
            collate_fn = PadSequenceManyToOne()
        else:
            collate_fn = None
        return collate_fn

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Splits the data for use in training, validation, and testing.
        Uses torch.utils.data.random_split() to split the data.
        Splitting is based on the test_size and test_seed.
        The test_size can be a float or an int.
        If a spotpython scaler object is defined, the data will be scaled.

        Args:
            stage (Optional[str]):
                The current stage. Can be "fit" (for training and validation), "test" (testing),
                or None (for all three stages). Defaults to None.

        Examples:
            >>> from spotpython.data.lightdatamodule import LightDataModule
                from spotpython.data.csvdataset import CSVDataset
                import torch
                dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
                data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
                data_module.setup()
                print(f"Training set size: {len(data_module.data_train)}")
                Training set size: 3

        """
        if self.data_full is not None:
            self._setup_full_data_provided(stage)
        elif self.data_val is not None:
            self._setup_val_data_provided(stage)
        else:
            self._setup_test_data_provided(stage)

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader, i.e., a pytorch DataLoader instance
        using the training dataset.

        Returns:
            DataLoader: The training dataloader.

        Examples:
            >>> from spotpython.data.lightdatamodule import LightDataModule
                from spotpython.data.csvdataset import CSVDataset
                import torch
                dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
                data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
                data_module.setup()
                print(f"Training set size: {len(data_module.data_train)}")
                Training set size: 3

        """
        if self.verbosity > 0:
            print(f"LightDataModule.train_dataloader(). data_train size: {len(self.data_train)}")
        # print(f"LightDataModule: train_dataloader(). batch_size: {self.batch_size}")
        # print(f"LightDataModule: train_dataloader(). num_workers: {self.num_workers}")
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=self.shuffle_train, collate_fn=self._get_collate_fn(collate_fn_name=self.collate_fn_name), num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader, i.e., a pytorch DataLoader instance
        using the validation dataset.

        Returns:
            DataLoader: The validation dataloader.

        Examples:
            >>> from spotpython.data.lightdatamodule import LightDataModule
                from spotpython.data.csvdataset import CSVDataset
                import torch
                dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
                data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
                data_module.setup()
                print(f"Training set size: {len(data_module.data_val)}")
                Training set size: 3
        """
        if self.verbosity > 0:
            print(f"LightDataModule.val_dataloader(). Val. set size: {len(self.data_val)}")
        # print(f"LightDataModule: val_dataloader(). batch_size: {self.batch_size}")
        # print(f"LightDataModule: val_dataloader(). num_workers: {self.num_workers}")
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=self.shuffle_val, collate_fn=self._get_collate_fn(collate_fn_name=self.collate_fn_name), num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader, i.e., a pytorch DataLoader instance
        using the test dataset.

        Returns:
            DataLoader: The test dataloader.

        Examples:
            >>> from spotpython.data.lightdatamodule import LightDataModule
                from spotpython.data.csvdataset import CSVDataset
                import torch
                dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
                data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
                data_module.setup()
                print(f"Test set size: {len(data_module.data_test)}")
                Test set size: 6

        """
        if self.verbosity > 0:
            print(f"LightDataModule.test_dataloader(). Test set size: {len(self.data_test)}")
        # print(f"LightDataModule: test_dataloader(). batch_size: {self.batch_size}")
        # print(f"LightDataModule: test_dataloader(). num_workers: {self.num_workers}")
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=self.shuffle_test, collate_fn=self._get_collate_fn(collate_fn_name=self.collate_fn_name), num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        """
        Returns the predict dataloader, i.e., a pytorch DataLoader instance
        using the predict dataset.

        Returns:
            DataLoader: The predict dataloader.

        Examples:
            >>> from spotpython.data.lightdatamodule import LightDataModule
                from spotpython.data.csvdataset import CSVDataset
                import torch
                dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
                data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
                data_module.setup()
                print(f"Predict set size: {len(data_module.data_predict)}")
                Predict set size: 6

        """
        if self.verbosity > 0:
            print(f"LightDataModule.predict_dataloader(). Predict set size: {len(self.data_predict)}")
        # print(f"LightDataModule: predict_dataloader(). batch_size: {self.batch_size}")
        # print(f"LightDataModule: predict_dataloader(). num_workers: {self.num_workers}")
        return DataLoader(self.data_predict, batch_size=len(self.data_predict), shuffle=False, collate_fn=self._get_collate_fn(collate_fn_name=self.collate_fn_name), num_workers=self.num_workers)
