import pytest
import torch
from torch.utils.data import TensorDataset
from spotpython.data.lightdatamodule import LightDataModule
from spotpython.utils.scaler import TorchStandardScaler

# Sample data for testing
@pytest.fixture
def sample_dataset():
    # Creating a simple dataset with 10 samples and 5 features each
    data = torch.randn(12, 5)
    targets = torch.randint(0, 2, (12,))
    return TensorDataset(data, targets)

# Fixture for the data module
@pytest.fixture
def data_module(sample_dataset):
    return LightDataModule(
        dataset=sample_dataset,
        batch_size=2,
        test_size=0.5,
        scaler=TorchStandardScaler(),
        verbosity=1
    )

def fit_scaler_for_test(data_module):
    # Manually fit the scaler on the full dataset before the test stage
    full_dataset = data_module.data_full
    full_data = torch.stack([full_dataset[i][0] for i in range(len(full_dataset))]).squeeze(1)
    data_module.scaler.fit(full_data)

def test_test_dataloader(data_module):
    # Fit the scaler using the entire dataset to ensure it's ready for transformations
    fit_scaler_for_test(data_module)
    # Now run setup for the "test" stage
    data_module.setup(stage="test")
    # Check the test dataloader
    test_loader = data_module.test_dataloader()
    batch = next(iter(test_loader))
    assert batch[0].shape[0] == 2  # Since batch size is set to 2

def test_scaling_and_transformation(data_module):
    data_module.setup(stage="fit")
    # Ensure scaler fits and transforms dataset correctly
    assert isinstance(data_module.data_train, TensorDataset)
    assert isinstance(data_module.data_val, TensorDataset)

def test_setup_splits_data_correctly(data_module):
    data_module.setup(stage=None)
    # Test the split sizes
    assert len(data_module.data_train) == 3  # As calculated from splits for a dataset of 10
    assert len(data_module.data_val) == 3
    assert len(data_module.data_test) == 6

def test_train_dataloader(data_module):
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    assert batch[0].shape[0] == 2  # batch size should match

def test_val_dataloader(data_module):
    data_module.setup(stage="fit")
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    assert batch[0].shape[0] == 2  # batch size should match
