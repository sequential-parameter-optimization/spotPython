import torch
from torch.utils.data import DataLoader
from spotpython.data.csvdataset import CSVDataset


def test_csv_dataset():
    # Create an instance of CSVDataset for testing
    dataset = CSVDataset(target_column="prognosis")

    # Test the length of the dataset
    assert len(dataset) > 0

    # Test __getitem__
    idx = 0
    sample = dataset[idx]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    feature, target = sample
    assert isinstance(feature, torch.Tensor)
    assert isinstance(target, torch.Tensor)

    # Test DataLoader
    batch_size = 3
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in dataloader:
        inputs, targets = batch
        assert inputs.size(0) == batch_size
        assert targets.size(0) == batch_size
        break
