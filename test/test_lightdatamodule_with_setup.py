import pytest
import torch
from torch.utils.data import TensorDataset
from lightning import seed_everything
from spotpython.data.lightdatamodule import LightDataModule

# Assuming LightDataModule is already imported from the provided code.

# Define a mock scaler for testing purpose.
class MockScaler:
    def fit(self, data):
        pass
    
    def transform(self, data):
        return data

# Define a simple dataset for testing.
def create_mock_dataset(size=12):
    data = torch.arange(size).float().view(-1, 1)
    target = torch.arange(size).long()
    return TensorDataset(data, target)

# Test initialization and data splits
@pytest.mark.parametrize("test_size, expected_train_size, expected_val_size, expected_test_size", [
    (0.5, 3, 3, 6),  # Split 12 items into 3 train, 3 val, 6 test
    (0.4, 5, 3, 5),  # Split 12 items into 5 train, 3 val, 5 test
])
def test_data_splitting(test_size, expected_train_size, expected_val_size, expected_test_size):
    dataset = create_mock_dataset()
    data_module = LightDataModule(dataset=dataset, batch_size=2, test_size=test_size, verbosity=1)
    data_module.setup()

    assert len(data_module.data_train) == expected_train_size
    assert len(data_module.data_val) == expected_val_size
    assert len(data_module.data_test) == expected_test_size


# Test DataLoader
def test_dataloader():
    dataset = create_mock_dataset()
    data_module = LightDataModule(dataset=dataset, batch_size=2, test_size=0.5, verbosity=1)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    assert len(train_loader.dataset) == len(data_module.data_train)
    assert len(val_loader.dataset) == len(data_module.data_val)
    assert len(test_loader.dataset) == len(data_module.data_test)

if __name__ == "__main__":
    pytest.main([__file__])