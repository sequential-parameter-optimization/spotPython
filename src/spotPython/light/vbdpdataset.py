import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    """
    A PyTorch Dataset for handling CSV data.

    Args:
        csv_file (str): The path to the CSV file. Defaults to "./data/VBDP/train.csv".
        train (bool): Whether the dataset is for training or not. Defaults to True.

    Attributes:
        data (Tensor): The data features.
        targets (Tensor): The data targets.
    """

    def __init__(
        self,
        csv_file: str = "./data/VBDP/train.csv",
        train: bool = True,
    ) -> None:
        super().__init__()
        self.csv_file = csv_file
        self.train = train
        self.data, self.targets = self._load_data()

    def _load_data(self) -> tuple:
        """
        Loads the data from the CSV file.

        Returns:
            tuple: A tuple containing the features and targets as tensors.
        Examples:
            >>> from spotPython.light import CSVDataset
            >>> dataset = CSVDataset()
            >>> print(dataset.data.shape)
            torch.Size([60000, 784])
            >>> print(dataset.targets.shape)
            torch.Size([60000])

        """
        data_df = pd.read_csv(self.csv_file)
        # drop the id column
        data_df = data_df.drop(columns=["id"])
        target_column = "prognosis"

        # Encode prognosis labels as integers
        label_encoder = LabelEncoder()
        targets = label_encoder.fit_transform(data_df[target_column])

        # Convert features to tensor
        features = data_df.drop(columns=[target_column]).values
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Convert targets to tensor
        targets_tensor = torch.tensor(targets, dtype=torch.long)
        return features_tensor, targets_tensor

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns the feature and target at the given index.

        Args:
            idx (int): The index.

        Returns:
            tuple: A tuple containing the feature and target at the given index.

        Examples:
            >>> from spotPython.light import CSVDataset
            >>> dataset = CSVDataset()
            >>> feature, target = dataset[0]
            >>> print(feature.shape)
            torch.Size([784])
            >>> print(target)
            tensor(0)

        """
        feature = self.data[idx]
        target = self.targets[idx]
        return feature, target

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        Examples:
            >>> from spotPython.light import CSVDataset
            >>> dataset = CSVDataset()
            >>> print(len(dataset))
            60000

        """
        return len(self.data)

    def extra_repr(self) -> str:
        """
        Returns a string representation of the dataset.

        Returns:
            str: A string representation of the dataset.

        Examples:
            >>> from spotPython.light import CSVDataset
            >>> dataset = CSVDataset()
            >>> print(dataset)
            Split: Train

        """
        split = "Train" if self.train else "Test"
        return f"Split: {split}"
