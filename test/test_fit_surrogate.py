import numpy as np
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import spot
from spotpython.utils.init import fun_control_init, design_control_init

def test_fit_surrogate():
    # number of initial points:
    ni = 0
    X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
    fun = Analytical().fun_sphere
    fun_control = fun_control_init(
        noise=False,
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        show_progress=True,
    )
    design_control = design_control_init(init_size=ni)
    S = spot.Spot(
        fun=fun,
        fun_control=fun_control,
        design_control=design_control,
    )
    S.initialize_design(X_start=X_start)
    S.update_stats()
    S.fit_surrogate()

    # Correlation matrix should be square and the same size as the number of points
    if S.surrogate.name == "kriging":
        # Old Kriging implementation
        assert S.surrogate.Psi.shape[0] == S.X.shape[0]
    else:
        # New Kriging implementation
        assert S.surrogate.Psi_.shape[0] == S.X.shape[0]

    # Check the prediction for a known input
    predicted_value = S.surrogate.predict(np.array([[0, 0]]))
    expected_value = np.array([1.49011612e-08])

    # Assert that the predicted value is approximately equal to the expected value
    np.testing.assert_allclose(predicted_value, expected_value, atol=1e-6,
                               err_msg="Prediction for input [[0, 0]] does not match the expected value.")

if __name__ == '__main__':
    test_fit_surrogate()