def calculate_data_split(test_size, full_size, verbosity=0, stage=None) -> tuple:
    """
    Calculates the split sizes for training, validation, and test datasets.

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
    """
    if isinstance(test_size, float):
        full_train_size = round(1.0 - test_size, 2)
        val_size = round(full_train_size * test_size, 2)
        train_size = round(full_train_size - val_size, 2)
    else:
        # test_size is considered an int, training size calculation directly based on it
        full_train_size = full_size - test_size
        val_size = int(full_train_size * test_size / full_size)
        train_size = full_train_size - val_size

    if verbosity > 0:
        print(f"stage: {stage}")
    if verbosity > 1:
        print(f"full_sizefull_train_size: {full_train_size}")
        print(f"full_sizeval_size: {val_size}")
        print(f"full_sizetrain_size: {train_size}")
        print(f"full_sizetest_size: {test_size}")

    return full_train_size, val_size, train_size, test_size
