import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_california_housing


class CaliforniaHousing(Dataset):
    """
    A PyTorch Dataset for regression. A toy data set from scikit-learn.
    Features:
        * MedInc median income in block group
        * HouseAge median house age in block group
        * AveRooms average number of rooms per household
        * AveBedrms average number of bedrooms per household
        * Population block group population
        * AveOccup average number of household members
        * Latitude block group latitude
        * Longitude block group longitude
    The target variable is the median house value for California districts,
    expressed in hundreds of thousands of Dollars ($100,000).
    Samples total: 20640, Dimensionality: 8, Features: real, Target: real 0.15 - 5.
    This dataset was derived from the 1990 U.S. census, using one row per census block group.
    A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data
    (a block group typically has a population of 600 to 3,000 people).

    Args:
        feature_type (torch.dtype): The data type of the features. Defaults to torch.float.
        target_type (torch.dtype): The data type of the targets. Defaults to torch.long.
        train (bool): Whether the dataset is for training or not. Defaults to True.
        n_samples (int): The number of samples of the dataset. Defaults to None, which means the entire dataset is used.

    Attributes:
        data (Tensor): The data features.
        targets (Tensor): The data targets.

    Examples:
        >>> from torch.utils.data import DataLoader
            from spotpython.data.diabetes import Diabetes
            import torch
            dataset = Diabetes(feature_type=torch.float32, target_type=torch.float32)
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
        feature_type: torch.dtype = torch.float,
        target_type: torch.dtype = torch.float,
        train: bool = True,
        n_samples: int = None,
    ) -> None:
        super().__init__()
        self.feature_type = feature_type
        self.target_type = target_type
        self.train = train
        self.n_samples = n_samples
        self.data, self.targets = self._load_data()

    def _load_data(self) -> tuple:
        """Loads the data from scikit-learn and returns the features and targets.

        Returns:
            tuple: A tuple containing the features and targets.

        Examples:
            >>> from spotpython.data.diabetes import Diabetes
                dataset = Diabetes()
                print(dataset.data.shape)
                print(dataset.targets.shape)
                torch.Size([442, 10])
                torch.Size([442])
        """
        feature_df, target_df = fetch_california_housing(return_X_y=True, as_frame=True)
        if self.n_samples is not None:
            feature_df = feature_df[: self.n_samples]
            target_df = target_df[: self.n_samples]
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
