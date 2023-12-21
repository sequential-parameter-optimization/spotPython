from spotPython.build.kriging import Kriging
import numpy as np
from math import erf

def test_interpolation_property():
    """
    Kriging is a interpolator, so the mean prediction
    should be equal to the training points:
    check if the difference between the mean prediction
    and the true value in the training points is smaller than 1e-6.
    """
    from spotPython.build.kriging import Kriging
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import linspace, arange
    rng = np.random.RandomState(1)
    X = linspace(start=0, stop=10, num=1_0).reshape(-1, 1)
    y = np.squeeze(X * np.sin(X))
    training_indices = rng.choice(arange(y.size), size=6, replace=False)
    X_train, y_train = X[training_indices], y[training_indices]
    S = Kriging(name='kriging', seed=124)
    S.fit(X_train, y_train)
    mean_prediction, std_prediction, s_ei = S.predict(X, return_val="all")
    assert np.allclose(mean_prediction[training_indices], y[training_indices], atol=1e-6)

def test_ei():
    """
    Test computation of expected improvement based on (3.8) in Forrester et al. (2008).
    """
    S = Kriging(name='kriging', seed=124)
    S.mean_cod_y = [0.0, 0.0, 0.0, 0.0, 0.0]
    # assert that the S.exp_imp(1.0, 0.0) is equal to 0.0,
    # because EI is zero when the std is zero
    assert 0.0 == S.exp_imp(1.0, 0.0)
    # assert that the S.exp_imp(0.0, 1.0) is equal to 1/sqrt(2 pi)
    # assert S.exp_imp(0.0, 1.0) == 1/np.sqrt(2*np.pi)
    # play safe and use np.allclose
    assert np.allclose(S.exp_imp(0.0, 1.0), 1/np.sqrt(2*np.pi), atol=1e-6)
    # assert S.exp_imp(1.0, 1.0) is correct based on (3.8) in Forrester et al. (2008)
    assert np.allclose(S.exp_imp(1.0, 1.0), -(0.5 + 0.5*erf(-1/np.sqrt(2))) + 1/np.sqrt(2*np.pi)*np.exp(-1/2), atol=1e-6)

def test_de_bounds():
    """
    Test if the bounds for the DE algorithm are set correctly.
    """
    S = Kriging(name='kriging', seed=124)
    S.set_de_bounds()
    assert S.de_bounds == [[-3, 2]]
    n = 10
    S = Kriging(name='kriging', seed=124, n_theta=n)
    S.set_de_bounds()
    assert len(S.de_bounds) == n
    n=2
    p=4
    S = Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True)
    S.set_de_bounds()
    assert len(S.de_bounds) == n+p
    S = Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=False)
    S.set_de_bounds()
    assert len(S.de_bounds) == n

def test_extract_from_bounds():
    n=2
    p=2
    S = Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=False)
    S.extract_from_bounds(np.array([1, 2, 3]))
    assert len(S.theta) == n

def test_optimize_model():
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    n=2
    p=2
    S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
    S.initialize_variables(nat_X, nat_y)
    S.set_variable_types()
    S.nat_to_cod_init()
    S.set_theta_values()
    S.initialize_matrices()
    S.set_de_bounds()
    new_theta_p_Lambda = S.optimize_model()
    assert len(new_theta_p_Lambda) == n+p+1
    # no noise, so Lambda is not considered
    S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=False)
    S.initialize_variables(nat_X, nat_y)
    S.set_variable_types()
    S.nat_to_cod_init()
    S.set_theta_values()
    S.initialize_matrices()
    S.set_de_bounds()
    new_theta_p_Lambda = S.optimize_model()
    assert len(new_theta_p_Lambda) == n+p
