import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class PKLDataset(Dataset):
    """
    A PyTorch Dataset for handling pickle (*.pkl) data.

    Args:
        pkl_file (str): The path to the pkl file. Defaults to "./data/spotPython/data.pkl".
        train (bool): Whether the dataset is for training or not. Defaults to True.

    Attributes:
        data (Tensor): The data features.
        targets (Tensor): The data targets.
    """

    def __init__(
        self,
        pkl_file: str = "./data/spotPython/data.pkl",
        feature_type: torch.dtype = torch.float,
        target_column: str = "y",
        target_type: torch.dtype = torch.long,
        train: bool = True,
        rmNA=True,
    ) -> None:
        super().__init__()
        self.pkl_file = pkl_file
        self.feature_type = feature_type
        self.target_type = target_type
        self.target_column = target_column
        self.train = train
        self.rmNA = rmNA
        self.data, self.targets = self._load_data()

    def _load_data(self) -> tuple:
        with open(self.pkl_file, "rb") as f:
            df = pd.read_pickle(f)
        # rm rows with NA
        if self.rmNA:
            df = df.dropna()
        # Apply LabelEncoder to string columns
        le = LabelEncoder()
        df = df.apply(lambda col: le.fit_transform(col) if col.dtypes == object else col)

        # Split DataFrame into feature and target DataFrames
        feature_df = df.drop(columns=[self.target_column])
        target_df = df[self.target_column]

        # Convert DataFrames to PyTorch tensors
        feature_tensor = torch.tensor(feature_df.values, dtype=self.feature_type)
        target_tensor = torch.tensor(target_df.values, dtype=self.target_type)

        return feature_tensor, target_tensor

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns the feature and target at the given index.

        Args:
            idx (int): The index.

        Returns:
            tuple: A tuple containing the feature and target at the given index.

        Examples:
            >>> from spotPython.light.pkldataset import pklDataset
                dataset = pklDataset(pkl_file='./data/spotPython/data.pkl', target_column='prognosis')
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
            >>> from spotPython.light import pklDataset
            >>> dataset = pklDataset()
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
            >>> from spotPython.light import pklDataset
            >>> dataset = pklDataset()
            >>> print(dataset)
            Split: Train

        """
        split = "Train" if self.train else "Test"
        return f"Split: {split}"