import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_diabetes


class Diabetes(Dataset):
    """
    A PyTorch Dataset for regression. A toy data set from scikit-learn.
    Ten baseline variables, age, sex, body mass index, average blood pressure,
    and six blood serum measurements were obtained for each of n = 442 diabetes patients,
    as well as the response of interest,
    a quantitative measure of disease progression one year after baseline.
    Number of Instances: 442
    Number of Attributes:First 10 columns are numeric predictive values.
    Target: Column 11 is a quantitative measure of disease progression one year after baseline.
    Attribute Information:
        * age age in years
        * sex
        * bmi body mass index
        * bp average blood pressure
        * s1 tc, total serum cholesterol
        * s2 ldl, low-density lipoproteins
        * s3 hdl, high-density lipoproteins
        * s4 tch, total cholesterol / HDL
        * s5 ltg, possibly log of serum triglycerides level
        * s6 glu, blood sugar level

    Args:
        feature_type (torch.dtype): The data type of the features. Defaults to torch.float.
        target_type (torch.dtype): The data type of the targets. Defaults to torch.long.
        train (bool): Whether the dataset is for training or not. Defaults to True.

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
        self, feature_type: torch.dtype = torch.float, target_type: torch.dtype = torch.float, train: bool = True
    ) -> None:
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
            >>> from spotpython.data.diabetes import Diabetes
                dataset = Diabetes()
                print(dataset.data.shape)
                print(dataset.targets.shape)
                torch.Size([442, 10])
                torch.Size([442])
        """
        feature_df, target_df = load_diabetes(return_X_y=True, as_frame=True)
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

    def get_names(self) -> list:
        """
        Returns the names of the features.

        Returns:
            list: A list containing the names of the features.

        Examples:
            >>> from spotpython.data.diabetes import Diabetes
                dataset = Diabetes()
                print(dataset.get_names())
                ["age", "sex", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]
        """
        return ["age", "sex", "bmi", "bp", "s1_tc", "s2_ldl", "s3_hdl", "s4_tch", "s5_ltg", "s6_glu"]
