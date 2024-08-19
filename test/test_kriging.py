from spotpython.build.kriging import Kriging
import numpy as np
from math import erf
from numpy import log, var


def test_interpolation_property():
    """
    Kriging is a interpolator, so the mean prediction
    should be equal to the training points:
    check if the difference between the mean prediction
    and the true value in the training points is smaller than 1e-6.
    """
    from spotpython.build.kriging import Kriging
    import numpy as np
    from numpy import linspace, arange

    rng = np.random.RandomState(1)
    X = linspace(start=0, stop=10, num=1_0).reshape(-1, 1)
    y = np.squeeze(X * np.sin(X))
    training_indices = rng.choice(arange(y.size), size=6, replace=False)
    X_train, y_train = X[training_indices], y[training_indices]
    S = Kriging(name="kriging", seed=124)
    S.fit(X_train, y_train)
    mean_prediction, std_prediction, s_ei = S.predict(X, return_val="all")
    assert np.allclose(mean_prediction[training_indices], y[training_indices], atol=1e-6)


def test_ei():
    """
    Test computation of expected improvement based on (3.8) in Forrester et al. (2008).
    """
    S = Kriging(name="kriging", seed=124)
    S.aggregated_mean_y = [0.0, 0.0, 0.0, 0.0, 0.0]
    # assert that the S.exp_imp(1.0, 0.0) is equal to 0.0,
    # because EI is zero when the std is zero
    assert 0.0 == S.exp_imp(1.0, 0.0)
    # assert that the S.exp_imp(0.0, 1.0) is equal to 1/sqrt(2 pi)
    # assert S.exp_imp(0.0, 1.0) == 1/np.sqrt(2*np.pi)
    # play safe and use np.allclose
    assert np.allclose(S.exp_imp(0.0, 1.0), 1 / np.sqrt(2 * np.pi), atol=1e-6)
    # assert S.exp_imp(1.0, 1.0) is correct based on (3.8) in Forrester et al. (2008)
    assert np.allclose(
        S.exp_imp(1.0, 1.0), -(0.5 + 0.5 * erf(-1 / np.sqrt(2))) + 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2), atol=1e-6
    )


def test_de_bounds():
    """
    Test if the bounds for the DE algorithm are set correctly.
    """
    S = Kriging(name="kriging", seed=124)
    S.set_de_bounds()
    assert S.de_bounds == [[-3, 2]]
    n = 10
    S = Kriging(name="kriging", seed=124, n_theta=n)
    S.set_de_bounds()
    assert len(S.de_bounds) == n
    n = 2
    p = 4
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=True)
    S.set_de_bounds()
    assert len(S.de_bounds) == n + p
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=False)
    S.set_de_bounds()
    assert len(S.de_bounds) == n


def test_extract_from_bounds():
    n = 2
    p = 2
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=True, noise=False)
    S.extract_from_bounds(np.array([1, 2, 3]))
    assert len(S.theta) == n


def test_optimize_model():
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    n = 2
    p = 2
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
    S.initialize_variables(nat_X, nat_y)
    S.set_variable_types()
    S.set_theta_values()
    S.initialize_matrices()
    S.set_de_bounds()
    new_theta_p_Lambda = S.optimize_model()
    assert len(new_theta_p_Lambda) == n + p + 1
    # no noise, so Lambda is not considered
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=True, noise=False)
    S.initialize_variables(nat_X, nat_y)
    S.set_variable_types()
    S.set_theta_values()
    S.initialize_matrices()
    S.set_de_bounds()
    new_theta_p_Lambda = S.optimize_model()
    assert len(new_theta_p_Lambda) == n + p


def test_update_log():
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    n = 2
    p = 2
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
    S.initialize_variables(nat_X, nat_y)
    S.set_variable_types()
    S.set_theta_values()
    S.initialize_matrices()
    S.set_de_bounds()
    _ = S.optimize_model()
    S.update_log()
    assert len(S.log["negLnLike"]) == 1
    assert len(S.log["theta"]) == n
    assert len(S.log["p"]) == p
    assert len(S.log["Lambda"]) == 1
    S.update_log()
    # now that we have log iterations two, there should be two entries in the log
    assert len(S.log["negLnLike"]) == 2
    assert len(S.log["theta"]) == 2 * n
    assert len(S.log["p"]) == 2 * p
    assert len(S.log["Lambda"]) == 2


def test_fit():
    nat_X = np.array([[1, 0], [1, 0]])
    nat_y = np.array([1, 2])
    S = Kriging()
    S.fit(nat_X, nat_y)
    assert S.Psi.shape == (2, 2)
    assert len(S.log["negLnLike"]) == 1


def test_initialize_variables():
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    S = Kriging()
    S.initialize_variables(nat_X, nat_y)
    assert S.nat_X.all() == nat_X.all()
    assert S.nat_y.all() == nat_y.all()
    assert S.nat_X.shape == (2, 2)
    assert S.nat_y.shape == (2,)


def test_set_variable_types():
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    n = 2
    p = 2
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
    S.initialize_variables(nat_X, nat_y)
    S.set_variable_types()
    assert S.var_type == ["num", "num"]
    assert S.var_type == ["num", "num"]
    assert S.num_mask.all()
    assert np.logical_not(S.factor_mask).all()
    assert np.logical_not(S.int_mask).all()
    assert S.ordered_mask.all()
    nat_X = np.array([[1, 2, 3], [4, 5, 6]])
    nat_y = np.array([1, 2])
    n = 3
    p = 1
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
    S.initialize_variables(nat_X, nat_y)
    S.set_variable_types()
    S.var_type
    assert S.var_type == ["num", "num", "num"]


def set_theta_values():
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    n = 2
    p = 2
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
    S.initialize_variables(nat_X, nat_y)
    S.set_variable_types()
    S.set_theta_values()
    assert S.theta.all() == np.array([0.0, 0.0]).all()
    nat_X = np.array([[1, 2], [3, 4]])
    nat_y = np.array([1, 2])
    # n is set to 3, but the number of columns of nat_X is 2
    n = 3
    p = 2
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
    S.initialize_variables(nat_X, nat_y)
    S.set_variable_types()
    snt = S.n_theta
    S.set_theta_values()
    # since snt == 3, it is not equal to S.n_theta, which is 2 because
    # of the correction in the set_theta_values method
    assert S.n_theta != snt


def test_initialize_matrices():
    nat_X = np.array([[1, 2], [3, 4], [5, 6]])
    nat_y = np.array([1, 2, 3])
    n = 3
    p = 1
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
    S.initialize_variables(nat_X, nat_y)
    S.set_variable_types()
    S.set_theta_values()
    S.initialize_matrices()
    # if var(self.nat_y) is > 0, then self.pen_val = self.n * log(var(self.nat_y)) + 1e4
    # else self.pen_val = self.n * var(self.nat_y) + 1e4
    assert S.pen_val == nat_X.shape[0] * log(var(S.nat_y)) + 1e4
    assert S.Psi.shape == (n, n)
    #
    # use a zero variance, then the penalty should be computed without log()
    nat_y = np.array([1, 1, 1])
    n = 3
    p = 1
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=True, noise=True)
    S.initialize_variables(nat_X, nat_y)
    S.set_variable_types()
    S.set_theta_values()
    S.initialize_matrices()
    # if var(self.nat_y) is > 0, then self.pen_val = self.n * log(var(self.nat_y)) + 1e4
    # else self.pen_val = self.n * var(self.nat_y) + 1e4
    assert S.pen_val == nat_X.shape[0] * (var(S.nat_y)) + 1e4
    assert S.Psi.shape == (n, n)


def test_fun_likelihood():
    from spotpython.build.kriging import Kriging
    import numpy as np

    nat_X = np.array([[0], [1]])
    nat_y = np.array([0, 1])
    n = 1
    p = 1
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=True, noise=False)
    S.initialize_variables(nat_X, nat_y)
    S.set_variable_types()
    S.set_theta_values()
    S.initialize_matrices()
    S.set_de_bounds()
    new_theta_p_Lambda = S.optimize_model()
    S.extract_from_bounds(new_theta_p_Lambda)
    S.build_Psi()
    S.build_U()
    assert S.negLnLike < 0


def test_likelihood():
    nat_X = np.array([[1], [2]])
    nat_y = np.array([5, 10])
    n = 2
    p = 1
    S = Kriging(name="kriging", seed=124, n_theta=n, n_p=p, optim_p=True, noise=False, theta_init_zero=True)
    S.initialize_variables(nat_X, nat_y)
    S.set_variable_types()
    S.set_theta_values()
    S.initialize_matrices()
    S.build_Psi()
    S.build_U()
    S.likelihood()
    # assert S.mu is close to 7.5 with a tolerance of 1e-6
    assert np.allclose(S.mu, 7.5, atol=1e-6)
    E = np.exp(1)
    sigma2 = E / (E**2 - 1) * (25 / 4 + 25 / 4 * E)
    # asssert S.SigmaSqr is close to sigma2 with a tolerance of 1e-6
    assert np.allclose(S.SigmaSqr, sigma2, atol=1e-6)
