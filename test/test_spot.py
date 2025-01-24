import numpy as np
from spotpython.spot import Spot
from spotpython.fun.objectivefunctions import Analytical
from spotpython.utils.init import fun_control_init, design_control_init, optimizer_control_init, surrogate_control_init
from scipy.optimize import shgo
from scipy.optimize import direct
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing

def test_suggest_new_X_with_different_bounds():
    nn = 3
    fun_sphere = Analytical().fun_sphere
    fun_control = fun_control_init(
        lower=np.array([-5, -5]),
        upper=np.array([5, 5]),
        n_points=nn,
    )
    design_control = design_control_init(init_size=10)
    optimizer_control = optimizer_control_init()
    surrogate_control = surrogate_control_init()

    S = Spot(
        fun=fun_sphere,
        fun_control=fun_control,
        design_control=design_control,
        optimizer_control=optimizer_control,
        surrogate_control=surrogate_control,
    )
    
    S.X = S.design.scipy_lhd(
        S.design_control["init_size"], lower=S.lower, upper=S.upper
    )
    S.y = S.fun(S.X)
    S.fit_surrogate()
    X0 = S.suggest_new_X()

    assert X0.size == S.n_points * S.k
    assert X0.ndim == 2
    assert X0.shape[0] == nn
    assert X0.shape[1] == 2
    assert np.all(X0 >= S.lower)
    assert np.all(X0 <= S.upper)

def test_suggest_new_X_with_different_init_size():
    nn = 3
    init_size = 20
    fun_sphere = Analytical().fun_sphere
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        n_points=nn,
    )
    design_control = design_control_init(init_size=init_size)
    optimizer_control = optimizer_control_init()
    surrogate_control = surrogate_control_init()

    S = Spot(
        fun=fun_sphere,
        fun_control=fun_control,
        design_control=design_control,
        optimizer_control=optimizer_control,
        surrogate_control=surrogate_control,
    )
    
    S.X = S.design.scipy_lhd(
        S.design_control["init_size"], lower=S.lower, upper=S.upper
    )
    S.y = S.fun(S.X)
    S.fit_surrogate()
    X0 = S.suggest_new_X()

    assert X0.size == S.n_points * S.k
    assert X0.ndim == 2
    assert X0.shape[0] == nn
    assert X0.shape[1] == 2
    assert np.all(X0 >= S.lower)
    assert np.all(X0 <= S.upper)

def test_suggest_new_X_with_different_n_points():
    nn = 5
    fun_sphere = Analytical().fun_sphere
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        n_points=nn,
    )
    design_control = design_control_init(init_size=10)
    optimizer_control = optimizer_control_init()
    surrogate_control = surrogate_control_init()

    S = Spot(
        fun=fun_sphere,
        fun_control=fun_control,
        design_control=design_control,
        optimizer_control=optimizer_control,
        surrogate_control=surrogate_control,
    )
    
    S.X = S.design.scipy_lhd(
        S.design_control["init_size"], lower=S.lower, upper=S.upper
    )
    S.y = S.fun(S.X)
    S.fit_surrogate()
    X0 = S.suggest_new_X()

    assert X0.size == S.n_points * S.k
    assert X0.ndim == 2
    assert X0.shape[0] == nn
    assert X0.shape[1] == 2
    assert np.all(X0 >= S.lower)
    assert np.all(X0 <= S.upper)
    def test_suggest_new_X_with_different_bounds():
        nn = 3
        fun_sphere = Analytical().fun_sphere
        fun_control = fun_control_init(
            lower=np.array([-5, -5]),
            upper=np.array([5, 5]),
            n_points=nn,
        )
        design_control = design_control_init(init_size=10)
        optimizer_control = optimizer_control_init()
        surrogate_control = surrogate_control_init()

        S = Spot(
            fun=fun_sphere,
            fun_control=fun_control,
            design_control=design_control,
            optimizer_control=optimizer_control,
            surrogate_control=surrogate_control,
        )
        
        S.X = S.design.scipy_lhd(
            S.design_control["init_size"], lower=S.lower, upper=S.upper
        )
        S.y = S.fun(S.X)
        S.fit_surrogate()
        X0 = S.suggest_new_X()

        assert X0.size == S.n_points * S.k
        assert X0.ndim == 2
        assert X0.shape[0] == nn
        assert X0.shape[1] == 2
        assert np.all(X0 >= S.lower)
        assert np.all(X0 <= S.upper)

def test_suggest_new_X_with_different_init_size():
    nn = 3
    init_size = 20
    fun_sphere = Analytical().fun_sphere
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        n_points=nn,
    )
    design_control = design_control_init(init_size=init_size)
    optimizer_control = optimizer_control_init()
    surrogate_control = surrogate_control_init()

    S = Spot(
        fun=fun_sphere,
        fun_control=fun_control,
        design_control=design_control,
        optimizer_control=optimizer_control,
        surrogate_control=surrogate_control,
    )
    
    S.X = S.design.scipy_lhd(
        S.design_control["init_size"], lower=S.lower, upper=S.upper
    )
    S.y = S.fun(S.X)
    S.fit_surrogate()
    X0 = S.suggest_new_X()

    assert X0.size == S.n_points * S.k
    assert X0.ndim == 2
    assert X0.shape[0] == nn
    assert X0.shape[1] == 2
    assert np.all(X0 >= S.lower)
    assert np.all(X0 <= S.upper)

def test_suggest_new_X_with_different_n_points():
    nn = 5
    fun_sphere = Analytical().fun_sphere
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        n_points=nn,
    )
    design_control = design_control_init(init_size=10)
    optimizer_control = optimizer_control_init()
    surrogate_control = surrogate_control_init()

    S = Spot(
        fun=fun_sphere,
        fun_control=fun_control,
        design_control=design_control,
        optimizer_control=optimizer_control,
        surrogate_control=surrogate_control,
    )
    
    S.X = S.design.scipy_lhd(
        S.design_control["init_size"], lower=S.lower, upper=S.upper
    )
    S.y = S.fun(S.X)
    S.fit_surrogate()
    X0 = S.suggest_new_X()

    assert X0.size == S.n_points * S.k
    assert X0.ndim == 2
    assert X0.shape[0] == nn
    assert X0.shape[1] == 2
    assert np.all(X0 >= S.lower)
    assert np.all(X0 <= S.upper)

def test_suggest_new_X_with_different_optimizers():
    nn = 3
    fun_sphere = Analytical().fun_sphere
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        n_points=nn,
    )
    design_control = design_control_init(init_size=10)
    surrogate_control = surrogate_control_init()

    optimizers = [dual_annealing, differential_evolution, direct, shgo]

    for optimizer_name in optimizers:
        optimizer_control = optimizer_control_init()

        S = Spot(
            fun=fun_sphere,
            fun_control=fun_control,
            design_control=design_control,
            optimizer=optimizer_name,
            optimizer_control=optimizer_control,
            surrogate_control=surrogate_control,
        )
        
        S.X = S.design.scipy_lhd(
            S.design_control["init_size"], lower=S.lower, upper=S.upper
        )
        S.y = S.fun(S.X)
        S.fit_surrogate()
        X0 = S.suggest_new_X()

        assert X0.shape[0] <= nn
        assert X0.shape[1] == 2
        assert np.all(X0 >= S.lower)
        assert np.all(X0 <= S.upper)
