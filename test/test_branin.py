def test_branin():
    """
    Test branin function
    """
    from spotpython.fun.objectivefunctions import Analytical
    import numpy as np

    pi = np.pi
    # some value, e.g. at 0,0:
    X_0 = np.array([[0, 0]])
    # there are 3 min values: disregarding rounding errors, they
    # should be the same for the following three points:
    X_1 = np.array([[-pi, 12.275], [pi, 2.275], [9.42478, 2.475]])
    # first of the three points should be identical to X_0, the
    # other two should be shifted by 10 in y-direction (plus and minus)
    X_2 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    fun = Analytical()
    y_0 = fun.fun_branin(X_0)
    y_1 = fun.fun_branin(X_1)
    y_2 = fun.fun_branin_factor(X_2)

    assert round(y_1[0], 2) == round(y_1[1], 2)
    assert round(y_1[0], 2) == round(y_1[2], 2)

    assert (y_2[0] == y_0)[0]
    assert (y_2[1] == y_0 + 10)[0]
    assert (y_2[2] == y_0 - 10)[0]
