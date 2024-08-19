from spotpython.data.csvdataset import CSVDataset
import pytest
import torch
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


def create_mock_csv(data: str, filename: str):
    with open(filename, "w") as f:
        f.write(data)


@pytest.fixture
def mock_csv_file(tmp_path):
    data = """id,feature1,feature2,feature3,y
              1,A,10.1,100.1,positive
              2,B,20.2,200.2,negative
              3,C,30.3,300.3,positive
              4,D,40.4,400.4,negative
              5,E,50.5,500.5,positive"""
    filename = tmp_path / "data.csv"
    create_mock_csv(data, filename)
    return filename


def test_csvdataset_remove_na(mock_csv_file):
    # Add a row with NA values
    data_with_na = """id,feature1,feature2,feature3,y
                     1,A,10.1,100.1,positive
                     2,B,20.2,200.2,negative
                     3,C,30.3,300.3,positive
                     4,D,,400.4,negative
                     5,E,50.5,500.5,positive"""
    temp_dir = mock_csv_file.parent
    filename_na = temp_dir / "data_with_na.csv"
    create_mock_csv(data_with_na, filename_na)

    dataset = CSVDataset(filename=filename_na, target_column="y", rmNA=True, oe=OrdinalEncoder(), le=LabelEncoder())
    assert len(dataset) == 4  # One row with NA should be removed
    assert dataset.data.shape[0] == 4  # Four rows left


def test_csvdataset_non_numerical_target(mock_csv_file):
    dataset = CSVDataset(
        filename=mock_csv_file, target_column="y", target_type=torch.long, oe=OrdinalEncoder(), le=LabelEncoder()
    )
    assert len(set(dataset.targets.tolist())) == 2  # There should be two unique target classes after label encoding


def test_csvdataset_len(mock_csv_file):
    dataset = CSVDataset(filename=mock_csv_file, target_column="y", oe=OrdinalEncoder(), le=LabelEncoder())
    assert len(dataset) == 5  # Check the correct length
