import pytest
import torch
from spotPython.data.lightdatamodule import LightDataModule
from spotPython.data.csvdataset import CSVDataset


def test_light_data_module():
    # Create an instance of CSVDataset for testing
    dataset = CSVDataset(target_column='prognosis', feature_type=torch.long)

    # Test the length of the dataset
    assert len(dataset) > 0

    data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
    data_module.setup()

    # Test the length of val and train: should be equal, because test_size=0.5
    assert len(data_module.data_train) ==  len(data_module.data_val)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
