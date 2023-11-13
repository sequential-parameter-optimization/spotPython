import pytest
from spotPython.utils.init import fun_control_init
from spotPython.hyperparameters.values import set_data_module
from spotPython.data.lightdatamodule import LightDataModule
from spotPython.data.csvdataset import CSVDataset
import torch


def test_set_data_module():
    fun_control = fun_control_init()
    dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
    dm = LightDataModule(dataset=dataset, batch_size=5, test_size=7)
    dm.setup()
    set_data_module(fun_control=fun_control,
                    data_module=dm)
    data_module = fun_control["data_module"]
    # if assinged correctly, the length of the data_test should be the same as the length of the dataset dm:
    assert len(dm.data_test) == len(data_module.data_test)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
