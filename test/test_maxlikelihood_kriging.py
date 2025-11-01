import numpy as np
import pytest
from types import SimpleNamespace
import spotpython.surrogate.kriging as kriging_mod
from spotpython.surrogate.kriging import Kriging

def test_max_likelihood_calls_objective_and_returns_de_result(monkeypatch):
    model = Kriging(method="regression", n_theta=2)

    calls = {"likelihood_calls": 0, "bounds": None}

    # Likelihood returns (negLnLike, Psi, U). Only first is used by objective.
    def fake_likelihood(x: np.ndarray):
        calls["likelihood_calls"] += 1
        return float(np.sum(x**2)), None, None

    monkeypatch.setattr(model, "likelihood", fake_likelihood, raising=True)

    def fake_de(*, func=None, bounds=None, **kwargs):
        # Capture bounds, evaluate objective at a chosen candidate
        calls["bounds"] = bounds
        best_x = np.array([0.0, 0.0, -6.0], dtype=float)
        fun = func(best_x)  # should call model.likelihood(best_x)
        return SimpleNamespace(x=best_x, fun=fun)

    # Patch the module-level DE used inside Kriging.max_likelihood
    monkeypatch.setattr(kriging_mod, "differential_evolution", fake_de, raising=True)

    bounds = [(-3.0, 2.0), (-3.0, 2.0), (-9.0, 0.0)]
    best_x, best_fun = model.max_likelihood(bounds)

    assert calls["bounds"] == bounds
    # likelihood should have been called exactly once with best_x
    assert calls["likelihood_calls"] == 1
    assert np.allclose(best_x, np.array([0.0, 0.0, -6.0]))
    # fun is sum of squares of best_x
    assert best_fun == pytest.approx(np.sum(best_x**2))


def test_max_likelihood_propagates_de_output_even_if_objective_not_used(monkeypatch):
    model = Kriging(method="regression", n_theta=2)

    # Ensure that even if DE doesn't call objective, return values are passed through
    def fake_de(*, func=None, bounds=None, **kwargs):
        return SimpleNamespace(x=np.array([1.0, -1.0, -3.0]), fun=-123.456)

    monkeypatch.setattr(kriging_mod, "differential_evolution", fake_de, raising=True)

    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-9.0, 0.0)]
    x, f = model.max_likelihood(bounds)

    assert np.allclose(x, np.array([1.0, -1.0, -3.0]))
    assert f == pytest.approx(-123.456)


def test_max_likelihood_passes_bounds_correctly(monkeypatch):
    model = Kriging(method="reinterpolation", n_theta=3)

    seen = {"bounds": None}

    # Minimal likelihood to satisfy objective
    monkeypatch.setattr(model, "likelihood", lambda x: (0.0, None, None), raising=True)

    def fake_de(*, func=None, bounds=None, **kwargs):
        seen["bounds"] = bounds
        return SimpleNamespace(x=np.zeros(len(bounds)), fun=func(np.zeros(len(bounds))))

    monkeypatch.setattr(kriging_mod, "differential_evolution", fake_de, raising=True)

    bounds = [(-2.0, 2.0), (-1.0, 1.0), (0.0, 3.0), (-9.0, -3.0)]  # 3 thetas + 1 lambda
    model.max_likelihood(bounds)

    assert seen["bounds"] == bounds