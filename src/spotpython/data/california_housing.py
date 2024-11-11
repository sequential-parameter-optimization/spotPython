import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_california_housing


class CaliforniaHousing(Dataset):
    """
    A PyTorch Dataset for regression. A toy data set from scikit-learn.
    Data Set Characteristics:
    * Number of Instances: 20640
    * Number of Attributes: 8 numeric, predictive attributes and the target
    * Attribute Information:
        - MedInc median income in block group
        - HouseAge median house age in block group
        - AveRooms average number of rooms per household
        - AveBedrms average number of bedrooms per household
        - Population block group population
        - AveOccup average number of household members
        - Latitude block group latitude
        - Longitude block group longitude
    * Missing Attribute Values: None
    * Target: The target variable is the median house value for California districts,
        expressed in hundreds of thousands of dollars ($100,000).
        This dataset was obtained from the StatLib repository:
        https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
        This dataset was derived from the 1990 U.S. census, using one row per census block group.
        A block group is the smallest geographical unit for which the U.S. Census Bureau publishes
        sample data (a block group typically has a population of 600 to 3,000 people).
        A household is a group of people residing within a home. Since the average number of rooms
        and bedrooms in this dataset are provided per household, these columns may take surprisingly
        large values for block groups with few households and many empty houses, such as vacation resorts.

    Args:
        feature_type (torch.dtype): The data type of the features. Defaults to torch.float.
        target_type (torch.dtype): The data type of the targets. Defaults to torch.long.
        train (bool): Whether the dataset is for training or not. Defaults to True.

    Attributes:
        data (Tensor): The data features.
        targets (Tensor): The data targets.

    Examples:
        >>> from torch.utils.data import DataLoader
            from spotpython.data.california_housing import CaliforniaHousing
            import torch
            dataset = CaliforniaHousing(feature_type=torch.float32, target_type=torch.float32)
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

    def __init__(self, feature_type: torch.dtype = torch.float, target_type: torch.dtype = torch.float, train: bool = True) -> None:
        super().__init__()
        self.feature_type = feature_type
        self.target_type = target_type
        self.train = train
        self.names = self.get_names()
        self.data, self.targets = self._load_data()

    def _load_data(self) -> tuple:
        """Loads the data from scikit-learn and returns the features and targets.

        Returns:
            tuple: A tuple containing the features and targets.

        Examples:
            >>> from spotpython.data.california_housing import CaliforniaHousing
                dataset = CaliforniaHousing()
                print(dataset.data.shape)
                print(dataset.targets.shape)
                torch.Size([20640, 8])
                torch.Size([20640])
        """
        feature_df, target_df = fetch_california_housing(return_X_y=True, as_frame=True)
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
            >>> from spotpython.data.california_housing import CaliforniaHousing
                dataset = CaliforniaHousing()
                print(dataset.data.shape)
                print(dataset.targets.shape)
                torch.Size([20640, 8])
                torch.Size([20640])
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
            >>> from spotpython.data.california_housing import CaliforniaHousing
                dataset = CaliforniaHousing()
                print(len(dataset))
                20640
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

    def get_names(self) -> list:
        """
        Returns the names of the features.

        Returns:
            list: A list containing the names of the features.

        Examples:
            >>> from spotpython.data.california_housing import CaliforniaHousing
                dataset = CaliforniaHousing()
                print(dataset.get_names())
                ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        """
        housing = fetch_california_housing()
        return housing.feature_names
