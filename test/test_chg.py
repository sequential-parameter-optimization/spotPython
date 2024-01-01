import numpy as np
from spotPython.fun.objectivefunctions import analytical
from spotPython.spot import spot

def test_chg():
    fun = analytical().fun_sphere
    lower = np.array([-1, -1])
    upper = np.array([1, 1])
    S = spot.Spot(fun=fun,
                lower = lower,
                upper= upper,
    )
    z0 = [1, 2, 3]
    new_val_1 = 4
    new_val_2 = 5
    index_1 = 0
    index_2 = 2
    S.chg(x=new_val_1, y=new_val_2, z0=z0, i=index_1, j=index_2)
    assert np.equal(z0, np.array([4, 2, 5])).all()