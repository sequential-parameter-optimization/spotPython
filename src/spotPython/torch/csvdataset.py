from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return the data at the given index
        return self.data.iloc[idx]
