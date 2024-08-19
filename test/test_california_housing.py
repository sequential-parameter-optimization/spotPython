from spotpython.data.california_housing import CaliforniaHousing


def test_california_housing():
    # Initialize the dataset
    dataset = CaliforniaHousing()

    # Expected outputs
    expected_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
    expected_length = 20640

    # Assert if get_names() returns the correct feature names
    assert dataset.get_names() == expected_names, "The feature names do not match the expected values."

    # Assert if the length of the dataset is correct
    assert len(dataset) == expected_length, "The length of the dataset does not match the expected value."
