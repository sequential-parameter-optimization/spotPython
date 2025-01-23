import numpy as np
import pytest
from spotpython.spot import Spot
from spotpython.fun.objectivefunctions import Analytical
from spotpython.utils.init import fun_control_init

def test_suggest_X0():
    # Setup
    nn = 3
    fun_sphere = Analytical().fun_sphere
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        n_points=nn,
    )
    
    spot_1 = Spot(
        fun=fun_sphere,
        fun_control=fun_control,
    )
    
    # (S-2) Initial Design:
    spot_1.X = spot_1.design.scipy_lhd(
        spot_1.design_control["init_size"], 
        lower=spot_1.lower, 
        upper=spot_1.upper
    )
    # (S-3): Eval initial design:
    spot_1.y = spot_1.fun(spot_1.X)
    spot_1.fit_surrogate()
    X0 = spot_1.suggest_new_X()
    
        
    # Assertions
    assert X0.size == spot_1.n_points * spot_1.k, "X0 size mismatch"
    assert X0.ndim == 2, "X0 should have 2 dimensions"
    assert X0.shape[0] == nn, "X0 first dimension should match n_points"
    assert X0.shape[1] == 2, "X0 second dimension should match the problem dimensionality"

if __name__ == "__main__":
    pytest.main([__file__])