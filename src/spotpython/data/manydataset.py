from torch.utils.data import Dataset
import torch
import pandas as pd
from typing import List, Optional, Union


class ManyToManyDataset(Dataset):
    """
    A PyTorch Dataset for many-to-many data.

    Args:
        df_list (List[pd.DataFrame]): List of pandas DataFrames.
        target (str): The target column name.
        drop (Optional[Union[str, List[str]]]): Column(s) to drop from the DataFrames. Default is None.
        dtype (torch.dtype): Data type for the tensors. Default is torch.float32.

    Attributes:
        data (List[pd.DataFrame]): List of pandas DataFrames with specified columns dropped.
        target (List[torch.Tensor]): List of target tensors.
        features (List[torch.Tensor]): List of feature tensors.

    Examples:
        >>> import pandas as pd
        >>> from spotpython.data.manydataset import ManyToManyDataset
        >>> df1 = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'target': [5, 6]})
        >>> df2 = pd.DataFrame({'feature1': [7, 8], 'feature2': [9, 10], 'target': [11, 12]})
        >>> dataset = ManyToManyDataset([df1, df2], target='target', drop='feature2')
        >>> len(dataset)
        2
        >>> dataset[0]
        (tensor([[1.],
                 [2.]]), tensor([5., 6.]))
    """

    def __init__(
        self,
        df_list: List[pd.DataFrame],
        target: str,
        drop: Optional[Union[str, List[str]]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        try:
            self.data = [df.drop(drop, axis=1) for df in df_list]
        except KeyError:
            self.data = df_list
        self.target = [torch.tensor(df[target].to_numpy(), dtype=dtype) for df in self.data]
        self.features = [torch.tensor(df.drop([target], axis=1).to_numpy(), dtype=dtype) for df in self.data]

    def __getitem__(self, index: int):
        x = self.features[index]
        y = self.target[index]
        return x, y

    def __len__(self) -> int:
        return len(self.data)


class ManyToOneDataset(Dataset):
    """
    A PyTorch Dataset for many-to-one data.

    Args:
        df_list (List[pd.DataFrame]): List of pandas DataFrames.
        target (str): The target column name.
        drop (Optional[Union[str, List[str]]]): Column(s) to drop from the DataFrames. Default is None.
        dtype (torch.dtype): Data type for the tensors. Default is torch.float32.

    Attributes:
        data (List[pd.DataFrame]): List of pandas DataFrames with specified columns dropped.
        target (List[torch.Tensor]): List of target tensors.
        features (List[torch.Tensor]): List of feature tensors.

    Examples:
        >>> import pandas as pd
        >>> from spotpython.data.manydataset import ManyToOneDataset
        >>> df1 = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'target': [5, 6]})
        >>> df2 = pd.DataFrame({'feature1': [7, 8], 'feature2': [9, 10], 'target': [11, 12]})
        >>> dataset = ManyToOneDataset([df1, df2], target='target', drop='feature2')
        >>> len(dataset)
        2
        >>> dataset[0]
        (tensor([[1.],
                 [2.]]), tensor(5.))
    """

    def __init__(
        self,
        df_list: List[pd.DataFrame],
        target: str,
        drop: Optional[Union[str, List[str]]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        try:
            self.data = [df.drop(drop, axis=1) for df in df_list]
        except KeyError:
            self.data = df_list
        self.target = [torch.tensor(df[target].to_numpy()[0], dtype=dtype) for df in self.data]
        self.features = [torch.tensor(df.drop([target], axis=1).to_numpy(), dtype=dtype) for df in self.data]

    def __getitem__(self, index: int):
        x = self.features[index]
        y = self.target[index]
        return x, y

    def __len__(self) -> int:
        return len(self.data)
