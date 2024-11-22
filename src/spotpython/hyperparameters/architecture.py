def generate_div2_list(n, n_min) -> list:
    result = []
    current = n
    repeats = 1
    max_repeats = 4
    while current >= n_min:
        result.extend([current] * min(repeats, max_repeats))
        current = current // 2
        repeats = repeats + 1
    return result


def get_hidden_sizes(_L_in, l1, n=10) -> list:
    """
    Generates a list of hidden sizes for a neural network with a given input size.
    Starting with size l1, the list is generated by dividing the input size by 2 until the minimum size is reached.

    Args:
        _L_in (int):
            input size.
        l1 (int):
            number of neurons in the first hidden layer.
        n (int):
            number of hidden sizes to generate.

    Returns:
        (list):
            list of hidden sizes.

    Examples:
        >>> from spotpython.hyperparameters.architecture import get_hidden_sizes
            _L_in = 10
            l1 = 20
            n = 4
            get_hidden_sizes(_L_in, l1, n)
            [20, 10, 10, 5]
    """
    if l1 < 4:
        raise ValueError("l1 must be at least 4")
    n_low = _L_in // 4
    n_high = max(l1, 2 * n_low)
    hidden_sizes = generate_div2_list(n_high, n_low)
    # keep only the first n values of hidden_sizes list
    if len(hidden_sizes) > n:
        hidden_sizes = hidden_sizes[:n]
    return hidden_sizes
