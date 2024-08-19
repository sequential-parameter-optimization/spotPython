import torch
from spotpython.data.lightdatamodule import LightDataModule
from spotpython.data.csvdataset import CSVDataset


def test_light_data_module():
    # Create an instance of CSVDataset for testing
    dataset = CSVDataset(target_column="prognosis", feature_type=torch.long)

    # Test the length of the dataset
    assert len(dataset) > 0

    data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
    data_module.setup()

    # Test the length of val and train: should be equal, because test_size=0.5
    assert len(data_module.data_train) == len(data_module.data_val)


def test_light_data_module_test_size():
    # Create an instance of CSVDataset for testing
    dataset = CSVDataset(target_column="prognosis", feature_type=torch.long)

    # Test the length of the dataset
    assert len(dataset) > 0

    # Now testing an absolute test_size
    data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=7)
    data_module.setup()

    # Test the length of val and train: should be equal, because test_size=0.5
    assert len(data_module.data_test) == 7


def test_light_data_module_sizes():
    # data.csv is simple csv file with 11 samples
    dataset = CSVDataset(csv_file="data.csv", target_column="prognosis", feature_type=torch.long)
    data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
    data_module.setup()
    print(f"Training set size: {len(data_module.data_train)}")
    print(f"Validation set size: {len(data_module.data_val)}")
    print(f"Test set size: {len(data_module.data_test)}")
    assert len(data_module.data_train) == 3
    assert len(data_module.data_val) == 3
    assert len(data_module.data_test) == 6
