from spotpython.utils.split import calculate_data_split
import pytest

def test_calculate_data_split_float():
    full_size = 100
    test_size = 0.2
    expected_full_train_size = 0.8
    expected_val_size = 0.16
    expected_train_size = 0.64

    result = calculate_data_split(test_size, full_size)

    assert result == (expected_full_train_size, expected_val_size, expected_train_size, test_size), \
           f"Result was {result}, expected {(expected_full_train_size, expected_val_size, expected_train_size, test_size)}"

def test_calculate_data_split_int():
    full_size = 100
    test_size = 20
    expected_full_train_size = 80
    expected_val_size = 16  # Calculated as 80 * 20 / 100
    expected_train_size = 64  # 80 - 16

    result = calculate_data_split(test_size, full_size)

    assert result == (expected_full_train_size, expected_val_size, expected_train_size, test_size), \
           f"Result was {result}, expected {(expected_full_train_size, expected_val_size, expected_train_size, test_size)}"

def test_calculate_data_split_verbosity():
    full_size = 100
    test_size = 0.2

    # Ideally, we'd capture the output here as well
    # For now, we just confirm it runs without error
    result = calculate_data_split(test_size, full_size, verbosity=2, stage='test')

    expected_full_train_size = 0.8
    expected_val_size = 0.16
    expected_train_size = 0.64

    assert result == (expected_full_train_size, expected_val_size, expected_train_size, test_size)

if __name__ == "__main__":
    pytest.main()