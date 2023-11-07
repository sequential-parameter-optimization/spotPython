import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class CSVDataset(Dataset):
    """
    A PyTorch Dataset for handling CSV data.

    Args:
        csv_file (str): The path to the CSV file. Defaults to "./data/spotPython/data.csv".
        train (bool): Whether the dataset is for training or not. Defaults to True.

    Attributes:
        data (Tensor): The data features.
        targets (Tensor): The data targets.
    """

    def __init__(
        self,
        csv_file: str = "./data/spotPython/data.csv",
        target_column: str = "y",
        target_type: str = "float",
        train: bool = True,
    ) -> None:
        super().__init__()
        self.csv_file = csv_file
        self.train = train
        self.data, self.targets = self._load_data(csv_file=csv_file, target_column=target_column)

    def _load_data(self, csv_file, target_column):
        data_df = pd.read_csv(self.csv_file)

        # Identify numerical and string columns
        numerical_columns = data_df.select_dtypes(include=[int, float]).columns
        string_columns = data_df.select_dtypes(include=[object]).columns

        # Convert numerical columns to float32 tensors
        numerical_features = data_df[numerical_columns].values
        numerical_features_tensor = torch.tensor(numerical_features, dtype=torch.float32)

        # Encode string columns as label-encoded long tensors
        label_encoder = LabelEncoder()
        string_features = data_df[string_columns].apply(label_encoder.fit_transform).values
        string_features_tensor = torch.tensor(string_features, dtype=torch.long)

        # Concatenate numerical and string tensors to create the features tensor
        features_tensor = torch.cat((numerical_features_tensor, string_features_tensor), dim=1)

        # Encode prognosis labels as integers
        targets = label_encoder.fit_transform(data_df[target_column])

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
