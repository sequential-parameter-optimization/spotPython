import torch
from torch.utils.data import Dataset


class DataFrameDataset(Dataset):
    def __init__(self, df, target_column, dtype_x=torch.float32, dtype_y=torch.float32):
        x = df.drop(target_column, axis=1).values
        y = df[target_column].values

        self.x_tensor = torch.tensor(x, dtype=dtype_x)
        self.y_tensor = torch.tensor(y, dtype=dtype_y)

    def __len__(self):
        return len(self.y_tensor)

    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]
