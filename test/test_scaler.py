import torch
from spotPython.data.lightdatamodule import LightDataModule
from spotPython.data.csvdataset import CSVDataset
from spotPython.utils.scaler import TorchStandardScaler
from spotPython.data.california_housing import CaliforniaHousing

def test_scaler():
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
    #assert that overall mean goes against zero
    assert overall_mean < 0.00001
    

