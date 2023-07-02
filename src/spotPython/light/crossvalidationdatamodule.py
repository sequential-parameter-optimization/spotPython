import lightning as L
from torch.utils.data import DataLoader, Subset
from typing import Optional
from spotPython.light.csvdataset import CSVDataset
from sklearn.model_selection import KFold
from torch.utils.data import Dataset


class CrossValidationDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        k: int = 1,
        split_seed: int = 42,
        num_splits: int = 10,
        DATASET_PATH: str = "./data",
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.DATASET_PATH = DATASET_PATH
        self.num_workers = num_workers
        self.k = k
        self.split_seed = split_seed
        self.num_splits = num_splits
        self.pin_memory = pin_memory
        self.save_hyperparameters(logger=False)
        assert 0 <= self.k < self.num_splits, "incorrect fold number"

        # no data transformations
        self.transforms = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):
        if not self.data_train and not self.data_val:
            dataset_full = CSVDataset(csv_file="./data/VBDP/train.csv", train=True)
            kf = KFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.split_seed)
            all_splits = [k for k in kf.split(dataset_full)]
            train_indexes, val_indexes = all_splits[self.hparams.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
            self.data_train = Subset(dataset_full, train_indexes)
            self.data_val = Subset(dataset_full, val_indexes)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
