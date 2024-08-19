import torch
from spotpython.data.lightdatamodule import LightDataModule
from spotpython.utils.scaler import TorchStandardScaler, TorchMinMaxScaler
from spotpython.data.california_housing import CaliforniaHousing


def test_standard_scaler():
    """
    Test if TorchStandardScaler scales data around 0.
    """
    dataset = CaliforniaHousing(feature_type=torch.float32, target_type=torch.float32)
    scaler = TorchStandardScaler()
    data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5, scaler=scaler)
    data_module.setup()

    loader = data_module.train_dataloader

    total_sum = None
    total_count = 0

    # Iterate over batches in the DataLoader
    for batch in loader():
        inputs, targets = batch
        if total_sum is None:
            total_sum = inputs.sum(dim=0)
        else:
            total_sum += inputs.sum(dim=0)
        total_count += inputs.shape[0]

    # Calculate the mean over all inputs
    mean_inputs = total_sum / total_count
    overall_mean = mean_inputs.mean()
    # assert that overall mean goes against zero
    assert overall_mean < 0.00001


def test_min_max_scaler():
    """
    Test if TorchMinMaxScaler scales data between 0 and 1.
    """
    dataset = CaliforniaHousing(feature_type=torch.float32, target_type=torch.float32)
    scaler = TorchMinMaxScaler()
    data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5, scaler=scaler)
    data_module.setup()

    loader = data_module.train_dataloader

    # Iterate over batches in the DataLoader
    for batch in loader():
        inputs, targets = batch
        assert torch.all(inputs >= 0) and torch.all(inputs <= 1), "Inputs are not scaled between 0 and 1"
