import lightning as L
from torch.utils.data import DataLoader, random_split

from spotPython.light.csvdataset import CSVDataset


class CSVDataModule(L.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = "./data", num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            data_full = CSVDataset(csv_file="./data/VBDP/train.csv", train=True)
            test_abs = int(len(data_full) * 0.6)
            self.data_train, self.data_val = random_split(data_full, [test_abs, len(data_full) - test_abs])

        # Assign test dataset for use in dataloader(s)
        # TODO: Adapt this to the VBDP Situation
        if stage == "test" or stage is None:
            self.data_test = CSVDataset(csv_file="./data/VBDP/train.csv", train=True)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)
