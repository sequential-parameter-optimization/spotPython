from spotpython.utils.transform import transform_multby2_int


def test_transform_multby2_int():
    assert transform_multby2_int(3) == 6
    assert transform_multby2_int(0) == 0
    assert transform_multby2_int(-5) == -10
    assert transform_multby2_int(100) == 200
