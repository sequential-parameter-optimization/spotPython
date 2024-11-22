import math
import warnings
from typing import List


def compute_lengths_from_fractions(fractions: List[float], dataset_length: int) -> List[int]:
    """Compute lengths of dataset splits from given fractions.

    Given a list of fractions that sum up to 1, compute the lengths of each
    corresponding partition of a dataset with a specified length. Each length is
    determined as `floor(frac * dataset_length)`. Any remaining items (due to flooring)
    are distributed among the partitions in a round-robin fashion.

    Args:
        fractions (List[float]): A list of fractions that should sum to 1.
        dataset_length (int): The length of the dataset.

    Returns:
        List[int]: A list of lengths corresponding to each fraction.

    Raises:
        ValueError: If the fractions do not sum to 1.
        ValueError: If any fraction is outside the range [0, 1].
        ValueError: If the sum of computed lengths does not equal the dataset length.

    Examples:
        >>> from spotpython.utils.split import compute_lengths_from_fractions
        >>> dataset_length = 5
        >>> fractions = [0.2, 0.3, 0.5]
        >>> compute_lengths_from_fractions(fractions, dataset_length)
        [1, 1, 3]

        In this example, 'dataset_length' is 5 and the 'fractions' specify the
        desired size distribution. The function calculates partitions of lengths
        [1, 1, 3] based on the given fractions.

    """
    if not math.isclose(sum(fractions), 1) or sum(fractions) > 1:
        raise ValueError("Fractions must sum up to 1.")

    subset_lengths: List[int] = []
    for i, frac in enumerate(fractions):
        if frac < 0 or frac > 1:
            raise ValueError(f"Fraction at index {i} is not between 0 and 1")
        n_items_in_split = int(math.floor(dataset_length * frac))
        subset_lengths.append(n_items_in_split)

    remainder = dataset_length - sum(subset_lengths)

    # Add 1 to all the lengths in a round-robin fashion until the remainder is 0
    for i in range(remainder):
        idx_to_add_at = i % len(subset_lengths)
        subset_lengths[idx_to_add_at] += 1

    lengths = subset_lengths
    for i, length in enumerate(lengths):
        if length == 0:
            warnings.warn(f"Length of split at index {i} is 0. " f"This might result in an empty dataset.")

    if sum(lengths) != dataset_length:
        raise ValueError("Sum of computed lengths does not equal the input dataset length!")

    return lengths


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
        if train_size + val_size + test_size != full_size:
            raise ValueError(f"full_size ({full_size}) != full_train_size ({full_train_size}) + test_size ({test_size})")

    if verbosity > 0:
        print(f"stage: {stage}")
    if verbosity > 1:
        print(f"full_sizefull_train_size: {full_train_size}")
        print(f"full_sizeval_size: {val_size}")
        print(f"full_sizetrain_size: {train_size}")
        print(f"full_sizetest_size: {test_size}")

    return full_train_size, val_size, train_size, test_size
