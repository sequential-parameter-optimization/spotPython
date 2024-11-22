def calculate_data_split(test_size, full_size, verbosity=0, stage=None) -> tuple:
    """
    Calculates the split sizes for training, validation, and test datasets.
    Returns a tuple containing the sizes (full_train_size, val_size, train_size, test_size),
    where full_train_size is the size of the full dataset minus the test set.

    Note:
        The first return value is full_train_size, i.e.,
        the size of the full dataset minus the test set.

    Args:
        test_size (float or int):
            The size of the test set.
            Can be a float for proportion or an int for absolute number of test samples.
        full_size (int):
            The size of the full dataset.
        verbosity (int, optional):
            The level of verbosity for debug output. Defaults to 0.
        stage (str, optional):
            The stage of setup, for debug output if needed.

    Returns:
        tuple: A tuple containing the sizes (full_train_size, val_size, train_size, test_size).

    Examples:
        >>> from spotpython.utils.split import calculate_data_split
            # Using proportion for test size
            calculate_data_split(0.2, 1000)
                (0.8, 0.16, 0.64, 0.2)
            # Using absolute number for test size
            calculate_data_split(200, 1000)
                (800, 160, 640, 200)

    Raises:
        ValueError: If the sizes are not correct, i.e., full_size != train_size + val_size + test_size.
    """
    if isinstance(test_size, float):
        full_train_size = round(1.0 - test_size, 2)
        val_size = round(full_train_size * test_size, 2)
        train_size = 1.0 - test_size - val_size
        # check if the sizes are correct, i.e., 1.0 = train_size + val_size + test_size
        if full_train_size + test_size != 1.0:
            raise ValueError(f"full_size ({full_size}) != full_train_size ({full_train_size}) + test_size ({test_size})")
    else:
        # test_size is considered an int, training size calculation directly based on it
        # everything is calculated as an int
        # return values are also ints
        # check if test_size does not exceed full_size
        if test_size > full_size:
            raise ValueError(f"test_size ({test_size}) > full_size ({full_size})")
        full_train_size = full_size - test_size
        val_size = int(full_train_size * test_size / full_size)
        train_size = full_train_size - val_size
        # check if the sizes are correct, i.e., full_size = train_size + val_size + test_size
        if full_train_size + test_size != full_size:
            raise ValueError(f"full_size ({full_size}) != full_train_size ({full_train_size}) + test_size ({test_size})")

    if verbosity > 0:
        print(f"stage: {stage}")
    if verbosity > 1:
        print(f"full_sizefull_train_size: {full_train_size}")
        print(f"full_sizeval_size: {val_size}")
        print(f"full_sizetrain_size: {train_size}")
        print(f"full_sizetest_size: {test_size}")

    return full_train_size, val_size, train_size, test_size
