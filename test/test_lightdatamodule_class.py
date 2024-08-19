import pytest
import torch
from spotpython.data.lightdatamodule import LightDataModule
from spotpython.data.csvdataset import CSVDataset


class TestLightDataModule:
    @pytest.fixture
    def setup_data_module(self):
        # Setup the dataset and data module as per the provided code snippet.
        # Mock the data.csv file content
        csv_content = """feature1,feature2,prognosis
                        1,2,0
                        3,4,1
                        5,6,0
                        7,8,1
                        9,10,0
                        11,12,1
                        13,14,0
                        15,16,1
                        17,18,0
                        19,20,1
                        21,22,0
                        23,24,1"""

        with open("data.csv", "w") as f:
            f.write(csv_content)

        dataset = CSVDataset(csv_file="data.csv", target_column="prognosis", feature_type=torch.long)
        data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.5)
        data_module.setup()
        return data_module

    def test_predict_set_size(self, setup_data_module):
        data_module = setup_data_module
        assert len(data_module.data_predict) == 6, "Expected predict set size to be 6,"
        f"but got {len(data_module.data_predict)}"
