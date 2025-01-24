def test_set_de_bounds():
    """
    Test _set_de_bounds
    """
    import numpy as np
    from spotpython.build import Kriging

    X_train = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y_train = np.array([1.0, 2.0, 3.0])

    # Without Noise:

    S = Kriging(name="kriging", seed=123, log_level=50, n_theta=1, noise=False, cod_type="norm")
    S.fit(X_train, y_train)
    res = [[S.min_theta, S.max_theta]]
    S._set_de_bounds()
    assert np.array_equal(S.de_bounds, res)

    # Return:

    # assume 1 theta, fixed p, no noise:

    new_theta_p_Lambda = np.array([-2.11625348])
    S._extract_from_bounds(new_theta_p_Lambda)

    res_theta = np.array([-2.11625348])
    assert np.array_equal(S.theta, res_theta)
    # p should be 2.0
    res_p = np.array([2.0])
    assert np.array_equal(S.p, res_p)
    # Lambda should be None
    assert S.Lambda is None

    # With Noise:

    S = Kriging(name="kriging", seed=123, log_level=50, n_theta=1, noise=True, cod_type="norm")
    S.fit(X_train, y_train)

    res = [[S.min_theta, S.max_theta], [S.min_Lambda, S.max_Lambda]]

    S._set_de_bounds()

    assert np.array_equal(S.de_bounds, res)

    # Return:

    # assume 1 theta, fixed p, noise:

    new_theta_p_Lambda = np.array([-2.11625348, 0.1234])
    S._extract_from_bounds(new_theta_p_Lambda)

    res_theta = np.array([-2.11625348])
    assert np.array_equal(S.theta, res_theta)
    # p should be 2.0
    res_p = np.array([2.0])
    assert np.array_equal(S.p, res_p)
    # Lambda should be 0.1234
    res_Lambda = 0.1234
    assert S.Lambda == res_Lambda

    # 2nd Test Series
    from spotpython.build import Kriging
    import numpy as np
    import copy
    from numpy import linspace
    from numpy import ones, zeros, log, var, float64
    from numpy import empty_like
    from numpy import array
    from spotpython.design.spacefilling import SpaceFilling

    # One-dim objective function
    ni = 11
    nat_X = linspace(start=0, stop=10, num=ni).reshape(-1, 1)
    nat_y = np.squeeze(nat_X + 1)
    S = Kriging(name="kriging", min_theta=-3, max_theta=2, seed=124)
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

    S._initialize_variables(nat_X, nat_y)
    S.theta = zeros(S.n_theta)
    # TODO: Currently not used:
    S.x0_theta = ones((S.n_theta,)) * S.n / (100 * S.k)
    S.p = ones(S.n_p) * 2.0

    S.pen_val = S.n * log(var(S.nat_y)) + 1e4
    S.negLnLike = None

    S.gen = SpaceFilling(k=S.k, seed=S.seed)

    # matrix related
    S.LnDetPsi = None
    S.Psi = zeros((S.n, S.n), dtype=float64)
    S.psi = zeros((S.n, 1))
    S.one = ones(S.n)
    S.mu = None
    S.U = None
    S.SigmaSqr = None
    S.Lambda = None
    # build_Psi() and build_U() are called in fun_likelihood
    S._set_de_bounds()
    # 1. check default (only theta):
    assert S.de_bounds[0][0] == S.min_theta
    assert S.de_bounds[0][1] == S.max_theta
    # 2. Check theta and p:
    S = Kriging(name="kriging", min_theta=-4, max_theta=5, optim_p=True, seed=124)

    S._set_de_bounds()
    assert S.de_bounds[0][0] == S.min_theta
    assert S.de_bounds[0][1] == S.max_theta
    assert S.de_bounds[1][0] == S.min_p
    assert S.de_bounds[1][1] == S.max_p

    # 3. Check theta, p, and Lambda:
    S = Kriging(name="kriging", min_theta=-4, max_theta=5, optim_p=True, noise=True, seed=124)

    S._set_de_bounds()
    assert S.de_bounds[0][0] == S.min_theta
    assert S.de_bounds[0][1] == S.max_theta
    assert S.de_bounds[1][0] == S.min_p
    assert S.de_bounds[1][1] == S.max_p
    assert S.de_bounds[2][0] == S.min_Lambda
    assert S.de_bounds[2][1] == S.max_Lambda

    # 3. Check theta and Lambda:
    S = Kriging(name="kriging", min_theta=-4, max_theta=5, optim_p=False, noise=True, seed=124)

    S._set_de_bounds()
    assert S.de_bounds[0][0] == S.min_theta
    assert S.de_bounds[0][1] == S.max_theta
    assert S.de_bounds[1][0] == S.min_Lambda
    assert S.de_bounds[1][1] == S.max_Lambda
