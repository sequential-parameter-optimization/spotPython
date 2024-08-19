from spotpython.build.kriging import Kriging
from numpy import power
import numpy as np


def test_is_any():
    nat_X = np.array([[0], [1]])
    nat_y = np.array([0, 1])
    n = 1
    p = 1
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=True, noise=False)
    S.initialize_variables(nat_X, nat_y)
    S.set_variable_types()
    S.set_theta_values()
    assert np.equal(S.__is_any__(power(10.0, S.theta), 0), False)
    assert np.equal(S.__is_any__(S.theta, 0), True)
