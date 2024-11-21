import pytest
import torch
from torch.utils.data import TensorDataset
from spotpython.data.lightdatamodule import LightDataModule

# Sample data for testing
@pytest.fixture
def sample_dataset():
    # Create a simple dataset with 12 samples and 5 features each
    data = torch.randn(12, 5)
    targets = torch.randint(0, 2, (12,))
    return TensorDataset(data, targets)

# Fixture for the data module
@pytest.fixture
def data_module_no_scaler(sample_dataset):
    return LightDataModule(
        dataset=sample_dataset,
        batch_size=2,
        test_size=0.5,
        scaler=None,  # No scaler is used
        verbosity=1
    )

def test_setup_without_scaler(data_module_no_scaler):
    # Run setup for the "fit" stage
    data_module_no_scaler.setup(stage="fit")

    # Ensure datasets are created correctly
    assert len(data_module_no_scaler.data_train) > 0
    assert len(data_module_no_scaler.data_val) > 0

    # Run setup for the "test" stage
    data_module_no_scaler.setup(stage="test")

    # Check that the test data loader can be created and used
    test_loader = data_module_no_scaler.test_dataloader()
    batch = next(iter(test_loader))
    assert batch[0].shape[0] == 2  # Since batch size is set to 2