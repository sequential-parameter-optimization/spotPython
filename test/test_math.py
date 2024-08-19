from spotpython.utils.math import generate_div2_list


def test_generate_div2_list():
    result = generate_div2_list(128, 32)
    assert result == [128, 64, 64, 32, 32, 32, 32]

    assert generate_div2_list(64, 128) == []

    result = generate_div2_list(64, 63)
    assert result == [64]
