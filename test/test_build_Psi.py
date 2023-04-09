def test_build_Psi():
    """
    Test build_Psi
    """
    from spotPython.build.kriging import Kriging
    import numpy as np
    import copy
    import matplotlib.pyplot as plt
    from numpy import append, ndarray, multiply, isinf, linspace, meshgrid, ravel, argmin
    from numpy import ones, zeros, arange, log, var, float64
    from numpy import spacing, empty_like
    from numpy import array
    from spotPython.design.spacefilling import spacefilling
    import spotPython
    from spotPython.fun.objectivefunctions import analytical
    from spotPython.spot import spot
    from spotPython.utils.repair import repair_non_numeric

    # number of points:
    ni = 7

    fun = analytical().fun_sphere
    lower = np.array([-1,-1])
    upper = np.array([1,1])
    design_control={"init_size": ni}
    surrogate_control={
                "infill_criterion": "y",
                "n_points": 1,
            }
    # Spot: to generate initial design
    S_spot = spot.Spot(fun=fun,
                lower = lower, 
                upper= upper, 
                fun_evals = 25, 
                noise = False,
                log_level = 50, 
                design_control=design_control, 
                surrogate_control=surrogate_control)

    X = S_spot.generate_design(size=S_spot.design_control["init_size"],
                               repeats=S_spot.design_control["repeats"],
                               lower=S_spot.lower, 
                               upper=S_spot.upper)
    X = repair_non_numeric(X, S_spot.var_type)
    # (S-3): Eval initial design:
    y = fun(X)
    S_spot.min_y = min(y)
    S_spot.min_X = X[argmin(y)]
    # Kriging:


    S = Kriging(name='kriging',
                seed=124, 
                n_theta=2, 
                noise=True, 
                cod_type="norm")
    S.nat_X = copy.deepcopy(X)
    S.nat_y = copy.deepcopy(y)
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
    S.int_mask = array(list(map(lambda x: x == "int", S.var_type)))
    S.ordered_mask = array(list(map(lambda x: x == "num" or x == "int" or x == "float", S.var_type)))
    S.nat_to_cod_init()
    S.theta = zeros(S.n_theta)
    # TODO: Currently not used:
    S.x0_theta = ones((S.n_theta,)) * S.n / (100 * S.k)
    S.p = ones(S.n_p) * 2.0
    S.pen_val = S.n * log(var(S.nat_y)) + 1e4
    S.negLnLike = None
    S.gen = spacefilling(k=S.k, seed=S.seed)
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
    S.set_de_bounds()
    if S.model_optimizer.__name__ == 'dual_annealing':
        result = S.model_optimizer(func=S.fun_likelihood,
                                        bounds=S.de_bounds)
    elif S.model_optimizer.__name__ == 'differential_evolution':
        result = S.model_optimizer(func=S.fun_likelihood,
                                        bounds=S.de_bounds,
                                        maxiter=S.model_fun_evals,
                                        seed=S.seed)
    elif S.model_optimizer.__name__ == 'direct':
        result = S.model_optimizer(func=S.fun_likelihood,
                                        bounds=S.de_bounds,
                                        # maxfun=S.model_fun_evals,
                                        eps=1e-2)
    elif S.model_optimizer.__name__ == 'shgo':
        result = S.model_optimizer(func=S.fun_likelihood,
                                        bounds=S.de_bounds)
    elif S.model_optimizer.__name__ == 'basinhopping':
        result = S.model_optimizer(func=S.fun_likelihood,
                                        x0=S.min_X)
    else:
        result = S.model_optimizer(func=S.fun_likelihood, bounds=S.de_bounds)
    # Finally, set new theta and p values and update the surrogate again
    # for new_theta_p_Lambda in de_results["x"]:
    new_theta_p_Lambda = result["x"]
    S.extract_from_bounds(new_theta_p_Lambda)
    S.build_Psi()
    assert S.Psi.shape[0] == ni
    assert S.Psi.shape[1] == ni
    assert (S.Psi == S.Psi.T).all()
    