import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from spotpython.pinns.plot.result import plot_result
import os

# Dummy data for tests
X_NP = np.linspace(0, 10, 20)
Y_NP = np.sin(X_NP)
X_DATA_NP = np.array([1, 3, 5, 7, 9])
Y_DATA_NP = np.sin(X_DATA_NP)
YH_NP = np.cos(X_NP) # Dummy prediction
CURRENT_STEP = 100
XP_NP = np.array([2, 4, 6, 8])

X_TORCH = torch.tensor(X_NP, dtype=torch.float32)
Y_TORCH = torch.tensor(Y_NP, dtype=torch.float32)
X_DATA_TORCH = torch.tensor(X_DATA_NP, dtype=torch.float32)
Y_DATA_TORCH = torch.tensor(Y_DATA_NP, dtype=torch.float32)
YH_TORCH = torch.tensor(YH_NP, dtype=torch.float32)
XP_TORCH = torch.tensor(XP_NP, dtype=torch.float32)

X_LIST = X_NP.tolist()
Y_LIST = Y_NP.tolist()
X_DATA_LIST = X_DATA_NP.tolist()
Y_DATA_LIST = Y_DATA_NP.tolist()
YH_LIST = YH_NP.tolist()
XP_LIST = XP_NP.tolist()


@pytest.fixture(autouse=True)
def mock_matplotlib(mocker):
    """Auto-used fixture to mock matplotlib functions."""
    mocker.patch("matplotlib.pyplot.show")
    mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("matplotlib.pyplot.figure")
    mocker.patch("matplotlib.pyplot.close")
    # Mock gca to return a mock object that has transAxes
    mock_ax = mocker.MagicMock()
    mock_ax.transAxes = mocker.MagicMock()
    mocker.patch("matplotlib.pyplot.gca", return_value=mock_ax)
    # Mock legend to return a mock object that has get_texts
    mock_legend = mocker.MagicMock()
    mock_legend.get_texts.return_value = [] # Return an empty list for texts
    mocker.patch("matplotlib.pyplot.legend", return_value=mock_legend)


def test_plot_result_runs_without_error_numpy(mock_matplotlib):
    """Test that plot_result runs without error with basic numpy inputs."""
    try:
        plot_result(
            x=X_NP, y=Y_NP, x_data=X_DATA_NP, y_data=Y_DATA_NP, yh=YH_NP,
            current_step=CURRENT_STEP, show_plot=False
        )
    except Exception as e:
        pytest.fail(f"plot_result raised an exception with numpy inputs: {e}")

def test_plot_result_runs_without_error_torch(mock_matplotlib):
    """Test that plot_result runs without error with basic torch tensor inputs."""
    try:
        plot_result(
            x=X_TORCH, y=Y_TORCH, x_data=X_DATA_TORCH, y_data=Y_DATA_TORCH, yh=YH_TORCH,
            current_step=CURRENT_STEP, show_plot=False
        )
    except Exception as e:
        pytest.fail(f"plot_result raised an exception with torch inputs: {e}")

def test_plot_result_runs_without_error_list(mock_matplotlib):
    """Test that plot_result runs without error with basic list inputs."""
    try:
        plot_result(
            x=X_LIST, y=Y_LIST, x_data=X_DATA_LIST, y_data=Y_DATA_LIST, yh=YH_LIST,
            current_step=CURRENT_STEP, show_plot=False
        )
    except Exception as e:
        pytest.fail(f"plot_result raised an exception with list inputs: {e}")


def test_plot_result_show_plot_true(mock_matplotlib):
    """Test that plt.show() is called when show_plot is True."""
    plot_result(
        x=X_NP, y=Y_NP, x_data=X_DATA_NP, y_data=Y_DATA_NP, yh=YH_NP,
        current_step=CURRENT_STEP, show_plot=True
    )
    plt.show.assert_called_once()
    plt.close.assert_not_called() # Should not be called if show_plot is True

def test_plot_result_show_plot_false(mock_matplotlib):
    """Test that plt.close() is called when show_plot is False."""
    plot_result(
        x=X_NP, y=Y_NP, x_data=X_DATA_NP, y_data=Y_DATA_NP, yh=YH_NP,
        current_step=CURRENT_STEP, show_plot=False
    )
    plt.close.assert_called_once()
    plt.show.assert_not_called()

def test_plot_result_save_path_provided(mock_matplotlib, tmp_path):
    """Test that plt.savefig() is called when save_path is provided."""
    save_file = tmp_path / "test_plot.png"
    plot_result(
        x=X_NP, y=Y_NP, x_data=X_DATA_NP, y_data=Y_DATA_NP, yh=YH_NP,
        current_step=CURRENT_STEP, show_plot=False, save_path=str(save_file)
    )
    plt.savefig.assert_called_once_with(str(save_file), dpi=300, bbox_inches='tight')
    plt.close.assert_called_once() # Should also close if not showing

def test_plot_result_with_xp_numpy(mock_matplotlib):
    """Test that plot_result runs with xp (collocation points) as numpy."""
    try:
        plot_result(
            x=X_NP, y=Y_NP, x_data=X_DATA_NP, y_data=Y_DATA_NP, yh=YH_NP,
            current_step=CURRENT_STEP, xp=XP_NP, show_plot=False
        )
    except Exception as e:
        pytest.fail(f"plot_result raised an exception with xp (numpy): {e}")

def test_plot_result_with_xp_torch(mock_matplotlib):
    """Test that plot_result runs with xp (collocation points) as torch tensor."""
    try:
        plot_result(
            x=X_TORCH, y=Y_TORCH, x_data=X_DATA_TORCH, y_data=Y_DATA_TORCH, yh=YH_TORCH,
            current_step=CURRENT_STEP, xp=XP_TORCH, show_plot=False
        )
    except Exception as e:
        pytest.fail(f"plot_result raised an exception with xp (torch): {e}")

def test_plot_result_with_custom_lims(mock_matplotlib):
    """Test that plot_result runs with custom xlims and ylims."""
    custom_xlims = (0, 5)
    custom_ylims = (-1, 1)
    try:
        plot_result(
            x=X_NP, y=Y_NP, x_data=X_DATA_NP, y_data=Y_DATA_NP, yh=YH_NP,
            current_step=CURRENT_STEP, xlims=custom_xlims, ylims=custom_ylims,
            show_plot=False
        )
    except Exception as e:
        pytest.fail(f"plot_result raised an exception with custom xlims/ylims: {e}")
    # Note: Verifying plt.xlim/ylim calls would require more specific mocking of plt or axes objects.
    # For now, we ensure it runs.

def test_plot_result_no_xp(mock_matplotlib):
    """Test that plot_result runs correctly when xp is None."""
    try:
        plot_result(
            x=X_NP, y=Y_NP, x_data=X_DATA_NP, y_data=Y_DATA_NP, yh=YH_NP,
            current_step=CURRENT_STEP, xp=None, show_plot=False
        )
    except Exception as e:
        pytest.fail(f"plot_result raised an exception when xp is None: {e}")

