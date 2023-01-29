def test_nat_to_cod_init():
    """
    Test nat_to_cod_init
    """
    from spotPython.build.kriging import Kriging
    import numpy as np
    import copy
    import matplotlib.pyplot as plt
    from numpy import append, ndarray, multiply, isinf, linspace, meshgrid, ravel
    from numpy import ones, zeros, arange
    from numpy import spacing, empty_like
    from numpy import array

    # One-dim objective function
    ni = 11
    nat_X = linspace(start=0, stop=10, num=ni).reshape(-1, 1)
    nat_y = np.squeeze(nat_X +1)
    S = Kriging(name='kriging',  seed=124)
    S.nat_X = copy.deepcopy(nat_X)
    S.nat_y = copy.deepcopy(nat_y)
    S.n = S.nat_X.shape[0]
    S.k = S.nat_X.shape[1]
    S.cod_X = empty_like(S.nat_X)
    S.cod_y = empty_like(S.nat_y)
    # assume all variable types are "num" if "num" is
    # specified once:
    if len(S.var_type) == 1:
        S.var_type = S.var_type * S.k
    S.num_mask = array(list(map(lambda x: x == "num", S.var_type)))
    S.factor_mask = array(list(map(lambda x: x == "factor", S.var_type)))

    S.nat_to_cod_init()
    assert S.nat_X.ndim == 2
    assert S.nat_X.shape[0] == ni
    assert S.nat_y.ndim == 1
    assert S.nat_y.shape[0] == ni
    #
    assert S.cod_X.ndim == 2
    assert S.cod_X.shape[0] == ni
    assert S.cod_y.ndim == 1
    assert S.cod_y.shape[0] == ni