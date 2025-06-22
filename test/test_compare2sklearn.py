import numpy as np
from spotpython.surrogate.kriging import Kriging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
y = np.squeeze(X * np.sin(X))
rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)


S = Kriging(name='kriging',  seed=123, log_level=50, cod_type="norm")
S.fit(X_train, y_train)
S_mean_prediction, S_std_prediction, S_ei = S.predict(X, return_val="all")
print(f"Mean prediction shape: {mean_prediction.shape}, Std prediction shape: {std_prediction.shape}")
print(f"S Mean prediction shape: {S_mean_prediction.shape}, S Std prediction shape: {S_std_prediction.shape}")
print(f"Mean predictions: {mean_prediction[:5]} vs {S_mean_prediction[:5]}")
print(f"Standard deviations: {std_prediction[:5]} vs {S_std_prediction[:5]}")

assert np.allclose(mean_prediction, S_mean_prediction, atol=.55), "Mean predictions do not match"
assert np.allclose(std_prediction, S_std_prediction, atol=.5), "Standard deviations do not match"