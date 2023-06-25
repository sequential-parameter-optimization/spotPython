from typing import Dict
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
        "10 - ten",
    ]

    def __init__(
        self,
        csv_file: str = "./data/VBDP/train.csv",
        train: bool = True,
    ) -> None:
        super().__init__()
        self.csv_file = csv_file
        self.train = train
        self.data, self.targets = self._load_data()

    def _load_data(self):
        data_df = pd.read_csv(self.csv_file)
        target_column = "prognosis"

        # Encode prognosis labels as integers
        label_encoder = LabelEncoder()
        targets = label_encoder.fit_transform(data_df[target_column])

        # Convert features to tensor
        features = data_df.drop(columns=[target_column]).values
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Convert targets to tensor
        targets_tensor = torch.tensor(targets, dtype=torch.long)
        return features_tensor, targets_tensor

    def __getitem__(self, idx):
        feature = self.data[idx]
        target = self.targets[idx]
        return feature, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train else "Test"
        return f"Split: {split}"

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}
