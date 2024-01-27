import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pathlib


class CSVDataset(Dataset):
    """
    A PyTorch Dataset for handling CSV data.

    Args:
        filename (str): The path to the CSV file. Defaults to "data.csv".
        directory (str): The path to the directory where the CSV file is stored. Defaults to None.
        feature_type (torch.dtype): The data type of the features. Defaults to torch.float.
        target_column (str): The name of the target column. Defaults to "y".
        target_type (torch.dtype): The data type of the targets. Defaults to torch.long.
        train (bool): Whether the dataset is for training or not. Defaults to True.
        rmNA (bool): Whether to remove rows with NA values or not. Defaults to True.
        dropId (bool): Whether to drop the "id" column or not. Defaults to False.
        **desc (Any): Additional keyword arguments.

    Attributes:
        data (Tensor): The data features.
        targets (Tensor): The data targets.

    Examples:
        >>> from torch.utils.data import DataLoader
            from spotPython.data.csvdataset import CSVDataset
            import torch
            dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
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
    """

    def __init__(
        self,
        filename: str = "data.csv",
        directory: None = None,
        feature_type: torch.dtype = torch.float,
        target_column: str = "y",
        target_type: torch.dtype = torch.long,
        train: bool = True,
        rmNA=True,
        dropId=False,
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
        # print(f"Loading data from {self.path}")
        df = pd.read_csv(self.path, index_col=False)
        # rm rows with NA
        if self.rmNA:
            df = df.dropna()
        if self.dropId:
            df = df.drop(columns=["id"])

        oe = OrdinalEncoder()
        # Apply LabelEncoder to string columns
        le = LabelEncoder()
        # df = df.apply(lambda col: le.fit_transform(col) if col.dtypes == object else col)

        # Split DataFrame into feature and target DataFrames
        feature_df = df.drop(columns=[self.target_column])
        feature_df = oe.fit_transform(feature_df)
        target_df = df[self.target_column]
        # only apply LabelEncoder to target column if it is a string
        if target_df.dtype == object:
            target_df = le.fit_transform(target_df)

        # Convert DataFrames to PyTorch tensors
        feature_tensor = torch.tensor(feature_df, dtype=self.feature_type)
        target_tensor = torch.tensor(target_df, dtype=self.target_type)

        return feature_tensor, target_tensor

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns the feature and target at the given index.

        Args:
            idx (int): The index.

        Returns:
            tuple: A tuple containing the feature and target at the given index.

        Examples:
            >>> from spotPython.light.csvdataset import CSVDataset
                dataset = CSVDataset(filename='./data/spotPython/data.csv', target_column='prognosis')
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
