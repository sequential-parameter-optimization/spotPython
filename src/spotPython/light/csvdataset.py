import torch
import pandas as pd
from torch.utils.data import Dataset


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
        self.target_column = target_column
        self.train = train
        self.data, self.targets = self._load_data()

    def _load_data(self) -> tuple:
        print(f"Loading data from {self.csv_file}")
        print(f"Target column: {self.target_column}")
        # read the csv file into a pandas dataframe and use every column as a feature
        df = pd.read_csv(self.csv_file, index_col=False)
        print(f"Data shape: {df.shape}")
        print(f"Data head:\n{df.head()}")
        print(f"Data describe:\n{df.describe()}")

        # Identify types
        numerical_cols = df.select_dtypes(include=[int, float]).columns
        string_cols = df.select_dtypes(include=[object]).columns

        # Convert numerical columns to torch.float
        for col in numerical_cols:
            df[col] = torch.tensor(df[col].values, dtype=torch.float)

        # Convert string columns to torch.long
        for col in string_cols:
            df[col] = torch.tensor(df[col].astype("category").cat.codes.values, dtype=torch.long)

        # Extract target column and convert to torch.tensor
        # target_column = torch.tensor(df[self.target_column].values, dtype=df[self.target_column].dtype)
        target_column_tensor = torch.tensor(
            df[self.target_column].values, dtype=torch.tensor(df[self.target_column].values).dtype
        )

        # Drop target column to get features
        features_df = df.drop(columns=[self.target_column])

        # Convert dataframe to tensor and return
        features_tensor = torch.tensor(features_df.values, dtype=torch.float)

        return features_tensor, target_column_tensor

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns the feature and target at the given index.

        Args:
            idx (int): The index.

        Returns:
            tuple: A tuple containing the feature and target at the given index.

        Examples:
            >>> from spotPython.light.csvdataset import CSVDataset
                dataset = CSVDataset(csv_file='./data/spotPython/data.csv', target_column='prognosis')
                print(dataset.data.shape)
                print(dataset.targets.shape)
                torch.Size([11, 65])
                torch.Size([11])
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
