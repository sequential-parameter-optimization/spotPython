import torch
from torch.utils.data import DataLoader
from spotpython.data.pkldataset import PKLDataset


def test_pkl_dataset():
    # Create an instance of PKLDataset for testing
    dataset = PKLDataset(target_column="prognosis")

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
