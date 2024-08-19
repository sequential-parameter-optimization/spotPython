import json


def one_hot_encode(strings) -> dict:
    """One hot encode a list of strings.
    Arguments:
        strings (list): List of strings to encode.
    Returns:
        dict: Dictionary of strings and their one hot encoded values.
    Examples:
        >>> one_hot_encode(['a', 'b', 'c'])
        {'a': [1, 0, 0], 'b': [0, 1, 0], 'c': [0, 0, 1]}
    """
    n = len(strings)
    encoding_dict = {}
    for i, string in enumerate(strings):
        one_hot_encoded_value = [0] * n
        one_hot_encoded_value[i] = 1
        encoding_dict[string] = one_hot_encoded_value
    return encoding_dict


def sum_encoded_values(strings, encoding_dict) -> int:
    """Sum the encoded values of a list of strings.

    Args:
        strings (list): List of strings to encode.
        encoding_dict (dict): Dictionary of strings and their one hot encoded values.

    Returns:
        int: Decimal value of the sum of the encoded values.

    Examples:
        >>> encoding_dict = {'a': [1, 0, 0], 'b': [0, 1, 0], 'c': [0, 0, 1]}
            sum_encoded_values(['a', 'b', 'c'], encoding_dict)
            7
            sum_encoded_values(['a', 'c'], encoding_dict)
            5
    """
    result = [0] * len(list(encoding_dict.values())[0])
    for string in strings:
        encoded_value = encoding_dict.get(string)
        if encoded_value:
            result = [sum(x) for x in zip(result, encoded_value)]
    decimal_result = 0
    for i, value in enumerate(result[::-1]):
        decimal_result += value * (2**i)
    return decimal_result


def get_one_hot(alg: str, hyper_param: str, d: dict = None, filename: str = "data.json") -> dict:
    """Get one hot encoded values for a hyper parameter of an algorithm.
    Arguments:
        alg (str): Name of the algorithm.
        hyper_param (str): Name of the hyper parameter.
        d (dict): Dictionary of algorithms and their hyperparameters.
        filename (str): Name of the file containing the dictionary.
    Returns:
        dict: Dictionary of hyper parameter values and their one hot encoded values.

    Examples:
        >>> alg = "HoeffdingAdaptiveTreeClassifier"
            hyper_param = "split_criterion"
            d = {
                "HoeffdingAdaptiveTreeClassifier": {
                    "split_criterion": ["gini", "info_gain", "hellinger"],
                    "leaf_prediction": ["mc", "nb", "nba"],
                    "bootstrap_sampling": ["0", "1"]
                    },
                    "HoeffdingTreeClassifier": {
                        "split_criterion": ["gini", "info_gain", "hellinger"],
                        "leaf_prediction": ["mc", "nb", "nba"],
                        "binary_split": ["0", "1"],
                        "stop_mem_management": ["0", "1"]
                    }
                }
            get_one_hot(alg, hyper_param, d)
            {'gini': [1, 0, 0], 'info_gain': [0, 1, 0], 'hellinger': [0, 0, 1]}
    """
    if d is None:
        with open(filename, "r") as f:
            d = json.load(f)
    values = d[alg][hyper_param]
    one_hot_encoded_values = one_hot_encode(values)
    return one_hot_encoded_values


def add_missing_elements(a: list, b: list) -> list:
    """Add missing elements from list a to list b.
    Arguments:
        a (list): List of elements to check.
        b (list): List of elements to add to.

    Returns:
        list: List of elements with missing elements from list a added.

    Examples:
        >>> a = [1, 4]
            b = [1, 2]
            add_missing_elements(a, b)
            [1, 2, 4]
    """
    for element in a:
        if element not in b:
            b.append(element)


def find_closest_key(integer_value: int, encoding_dict: dict) -> str:
    """
    Given an integer value and an encoding dictionary that maps keys to binary values,
    this function finds the key in the dictionary whose binary value is closest to the binary
    representation of the integer value.

    Arguments:
        integer_value (int): The integer value to find the closest key for.
        encoding_dict (dict): The encoding dictionary that maps keys to binary values.

    Returns:
        str: The key in the encoding dictionary whose binary value is
        closest to the binary representation of the integer value.

    Examples:
        >>> encoding_dict = {'A': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1]}
            find_closest_key(6, encoding_dict)
            'B'
    """
    binary_value = [int(x) for x in format(integer_value, f"0{len(list(encoding_dict.values())[0])}b")]
    min_distance = float("inf")
    closest_key = None
    for key, encoded_value in encoding_dict.items():
        distance = sum([x != y for x, y in zip(binary_value, encoded_value)])
        if distance < min_distance:
            min_distance = distance
            closest_key = key
    return closest_key
