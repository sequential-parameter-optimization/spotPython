def generate_div2_list(n, n_min) -> list:
    """
    Generate a list of numbers from n to n_min (inclusive) by dividing n by 2
    until the result is less than n_min.
    This function starts with n and keeps dividing it by 2 until n_min is reached.
    The number of times each value is added to the list is determined by n // current.

    Args:
        n (int): The number to start with.
        n_min (int): The minimum number to stop at.

    Returns:
        list: A list of numbers from n to n_min (inclusive).

    Examples:
        from spotpython.utils.math import generate_div2_list
        generate_div2_list(10, 1)
        [10, 5, 5, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        generate_div2_list(10, 2)
        [10, 5, 5, 2, 2, 2, 2, 2]
    """
    result = []
    current = n
    while current >= n_min:
        result.extend([current] * (n // current))
        current = current // 2
    return result
