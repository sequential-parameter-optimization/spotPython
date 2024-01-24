from spotPython.utils.math import generate_list

def test_generate_list():
    # Test case 1: n = 10, n_min = 2
    result = generate_list(10, 2)
    assert result == [10, 5, 2]

    # Test case 2: n = 10, n_min = 3
    result = generate_list(10, 3)
    assert result == [10, 5]

    # Test case 3: n = 10, n_min = 5
    result = generate_list(10, 5)
    assert result == [10]

    # Test case 4: n = 10, n_min = 10
    result = generate_list(10, 10)
    assert result == []

    # Test case 5: n = 20, n_min = 2
    result = generate_list(20, 2)
    assert result == [20, 10, 5, 2]

    # Test case 6: n = 5, n_min = 2
    result = generate_list(5, 2)
    assert result == [5, 2]

    # Test case 7: n = 100, n_min = 10
    result = generate_list(100, 10)
    assert result == [100, 50, 25, 12, 6, 3]

    # Test case 8: n = 1, n_min = 1
    result = generate_list(1, 1)
    assert result == [1]