def test_build_psi_vec():
    """
    Test build_psi_vec, based on (2.34) in [Forrester et al. (2008)]
    """
    import numpy as np
    from spotPython.build.kriging import Kriging

    X_train = np.array([[1., 2.],
                        [2., 4.],
                        [3., 6.]])
    y_train = np.array([1., 2., 3.])

    S = Kriging(name='kriging',
                seed=123,
                log_level=50,
                n_theta=1,
                noise=False,
                cod_type="norm")
    S.fit(X_train, y_train)

    # force theta to simple values:
    S.theta = np.array([0.0])
    cod_x = np.array([1., 0.])
    S.psi = np.zeros((S.n, 1))
    S.build_psi_vec(cod_x)
    res = np.array([[np.exp(-4)],
        [np.exp(-17)],
        [np.exp(-40)]])

    assert np.array_equal(S.psi, res)
