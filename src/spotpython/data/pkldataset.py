import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pathlib


class PKLDataset(Dataset):
    """
    A PyTorch Dataset Class for handling pickle (*.pkl) data.

    Args:
        filename (str):
            The filename of the pkl file. Defaults to "data.pkl".
        directory (str):
            The directory where the pkl file is located. Defaults to None.
        feature_type (torch.dtype):
            The data type of the features. Defaults to torch.float.
        target_column (str):
            The name of the target column. Defaults to "y".
        target_type (torch.dtype):
            The data type of the targets. Defaults to torch.long.
        train (bool):
            Whether the dataset is for training or not. Defaults to True.
        rmNA (bool):
            Whether to remove rows with NA values or not. Defaults to True.
        **desc (Any):
            Additional arguments to be passed to the base class.

    Attributes:
        filename (str):
            The filename of the pkl file.
        directory (str):
            The directory where the pkl file is located.
        feature_type (torch.dtype):
            The data type of the features.
            Defaults to torch.float.
        target_column (str):
            The name of the target column.
        target_type (torch.dtype):
            The data type of the targets.
            Defaults to torch.float.
        train (bool):
            Whether the dataset is for training or not.
        rmNA (bool):
            Whether to remove rows with NA values or not.
        data (torch.Tensor):
            The features.
        targets (torch.Tensor):
            The targets.

    Notes:
        * `spotpython` comes with a sample pkl file, which is located at `spotpython/data/pkldataset.pkl`.

    Examples:
        >>> from spotpython.data.pkldataset import PKLDataset
            import torch
            from torch.utils.data import DataLoader
            dataset = PKLDataset(target_column='prognosis', feature_type=torch.long)
            # Set batch size for DataLoader
            batch_size = 5
            # Create DataLoader
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            # Iterate over the data in the DataLoader
            for batch in dataloader:
                inputs, targets = batch
                print(f"Batch Size: {inputs.size(0)}")
                print("---------------")
                print(f"Inputs: {inputs}")
                print(f"Targets: {targets}")
                break
            Batch Size: 5
            ---------------
            Inputs: tensor([[1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                    0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                    0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0,
                    1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                    0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            Targets: tensor([ 0,  1,  6,  9, 10])
        >>> # Load the data from a different directory:
        >>> # Similar to the above example, but with a different target column, full path, and different data type
        >>> from spotpython.data.pkldataset import PKLDataset
            import torch
            from torch.utils.data import DataLoader
            dataset = PKLDataset(directory="/Users/bartz/workspace/spotpython/notebooks/data/spotpython/",
                                filename="data_sensitive.pkl",
                                target_column='N',
                                feature_type=torch.float32,
                                target_type=torch.float32,
                                rmNA=True)
    """

    def __init__(
        self,
        filename: str = "data.pkl",
        directory: None = None,
        feature_type: torch.dtype = torch.float,
        target_column: str = "y",
        target_type: torch.dtype = torch.float,
        train: bool = True,
        rmNA=True,
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
        # ensure that self.target_type and self.feature_type are the same torch types
        if self.target_type != self.feature_type:
            raise ValueError("target_type and feature_type must be the same torch type")
        with open(self.path, "rb") as f:
            df = pd.read_pickle(f)
        # rm rows with NA
        if self.rmNA:
            df = df.dropna()

        # Split DataFrame into feature and target DataFrames
        feature_df = df.drop(columns=[self.target_column])

        # Identify non-numerical columns in the feature DataFrame
        non_numerical_columns = feature_df.select_dtypes(exclude=["number"]).columns.tolist()

        # Apply OrdinalEncoder to non-numerical feature columns
        if non_numerical_columns:
            feature_df[non_numerical_columns] = self.oe.fit_transform(feature_df[non_numerical_columns])

        target_df = df[self.target_column]

        # Check if the target column is non-numerical using dtype
        if not pd.api.types.is_numeric_dtype(target_df):
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
            >>> from spotpython.data.pkldataset import PKLDataset
                import torch
                from torch.utils.data import DataLoader
                dataset = PKLDataset(target_column='prognosis', feature_type=torch.long)
                print(dataset.data.shape)
                print(dataset.targets.shape)
                torch.Size([11, 64])
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
            >>> from spotpython.data.pkldataset import PKLDataset
                import torch
                from torch.utils.data import DataLoader
                dataset = PKLDataset(target_column='prognosis', feature_type=torch.long)
                print(len(dataset))
                11
        """
        return len(self.data)

    def extra_repr(self) -> str:
        """
        Returns a string with the filename and directory of the dataset.

        Returns:
            str: A string with the filename and directory of the dataset.

        Examples:
            >>> from spotpython.data.pkldataset import PKLDataset
                import torch
                from torch.utils.data import DataLoader
                dataset = PKLDataset(target_column='prognosis', feature_type=torch.long)
                print(dataset)
        """
        return "filename={}, directory={}".format(self.filename, self.directory)

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
