import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pathlib


class CSVDataset(Dataset):
    """
    A PyTorch Dataset for handling CSV data.
    """

    def __init__(
        self,
        filename: str = "data.csv",
        directory: None = None,
        feature_type: torch.dtype = torch.float,
        target_column: str = "y",
        target_type: torch.dtype = torch.float,
        train: bool = True,
        rmNA=True,
        dropId=False,
        oe=OrdinalEncoder(),
        le=LabelEncoder(),
        **desc,
    ) -> None:
        super().__init__()
        self.filename = filename
        self.directory = directory
        self.feature_type = feature_type
        self.target_type = target_type
        self.target_column = target_column
        self.train = train
        self.rmNA = rmNA
        self.dropId = dropId
        self.oe = oe
        self.le = le
        self.data, self.targets = self._load_data()

    @property
    def path(self):
        if self.directory:
            return pathlib.Path(self.directory).joinpath(self.filename)
        return pathlib.Path(__file__).parent.joinpath(self.filename)

    @property
    def _repr_content(self):
        content = super()._repr_content
        content["Path"] = str(self.path)
        return content

    def _load_data(self) -> tuple:
        df = pd.read_csv(self.path, index_col=False)

        # Remove rows with NA if specified
        if self.rmNA:
            df = df.dropna()

        # Drop the id column if specified
        if self.dropId and "id" in df.columns:
            df = df.drop(columns=["id"])

        # Split DataFrame into feature and target DataFrames
        feature_df = df.drop(columns=[self.target_column])

        # Identify non-numerical columns in the feature DataFrame
        non_numerical_columns = feature_df.select_dtypes(exclude=["number"]).columns.tolist()

        # Apply OrdinalEncoder to non-numerical feature columns
        if non_numerical_columns:
            if self.oe is None:
                raise ValueError(f"\n!!! non_numerical_columns in data: {non_numerical_columns}" "\nOrdinalEncoder object oe must be provided for encoding non-numerical columns")
            feature_df[non_numerical_columns] = self.oe.fit_transform(feature_df[non_numerical_columns])

        target_df = df[self.target_column]

        # Check if the target column is non-numerical using dtype
        if not pd.api.types.is_numeric_dtype(target_df):
            if self.le is None:
                raise ValueError(f"\n!!! The target column '{self.target_column}' is non-numerical" "\nLabelEncoder object le must be provided for encoding non-numerical target")
            target_df = self.le.fit_transform(target_df)

        # Convert DataFrames to NumPy arrays and then to PyTorch tensors
        feature_array = feature_df.to_numpy()
        target_array = target_df

        feature_tensor = torch.tensor(feature_array, dtype=self.feature_type)
        target_tensor = torch.tensor(target_array, dtype=self.target_type)

        return feature_tensor, target_tensor

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns the feature and target at the given index.

        Args:
            idx (int): The index.

        Returns:
            tuple: A tuple containing the feature and target at the given index.

        Examples:
            >>> from spotpython.light.csvdataset import CSVDataset
                dataset = CSVDataset(filename='./data/spotpython/data.csv', target_column='prognosis')
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
            >>> from spotpython.light import CSVDataset
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
            >>> from spotpython.light import CSVDataset
            >>> dataset = CSVDataset()
            >>> print(dataset)
            Split: Train

        """
        split = "Train" if self.train else "Test"
        return f"Split: {split}"

    def __ncols__(self) -> int:
        """
        Returns the number of columns in the dataset.

        Returns:
            int: The number of columns in the dataset.

        Examples:
            >>> from spotpython.data.pkldataset import PKLDataset
                import torch
                from torch.utils.data import DataLoader
                dataset = PKLDataset(target_column='prognosis', feature_type=torch.long)
                print(dataset.__ncols__())
                64
        """
        return self.data.size(1)
