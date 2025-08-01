import numpy as np
import pytest
import matplotlib
import os

from spotpython.surrogate.plot import plot_3d_contour

@pytest.fixture
def simple_plot_data():
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    return {
        "X_combined": X,
        "Y_combined": Y,
        "Z_combined": Z,
        "min_z": Z.min(),
        "max_z": Z.max(),
    }

def test_plot_3d_contour_runs_no_var_name(simple_plot_data, monkeypatch):
    monkeypatch.setattr("pylab.show", lambda: None)
    plot_3d_contour(
        simple_plot_data["X_combined"],
        simple_plot_data["Y_combined"],
        simple_plot_data["Z_combined"],
        simple_plot_data["min_z"],
        simple_plot_data["max_z"],
        show=True
    )
    plot_3d_contour(
        simple_plot_data["X_combined"],
        simple_plot_data["Y_combined"],
        simple_plot_data["Z_combined"],
        simple_plot_data["min_z"],
        simple_plot_data["max_z"],
        show=False
    )

def test_plot_3d_contour_with_var_name(simple_plot_data, monkeypatch):
    monkeypatch.setattr("pylab.show", lambda: None)
    plot_3d_contour(
        simple_plot_data["X_combined"],
        simple_plot_data["Y_combined"],
        simple_plot_data["Z_combined"],
        simple_plot_data["min_z"],
        simple_plot_data["max_z"],
        var_name=["x", "y"],
        show=True
    )

def test_plot_3d_contour_with_title_and_levels(simple_plot_data, monkeypatch):
    monkeypatch.setattr("pylab.show", lambda: None)
    plot_3d_contour(
        simple_plot_data["X_combined"],
        simple_plot_data["Y_combined"],
        simple_plot_data["Z_combined"],
        simple_plot_data["min_z"],
        simple_plot_data["max_z"],
        title="Test Title",
        contour_levels=5,
        show=True
    )

def test_plot_3d_contour_saves_file(simple_plot_data, tmp_path, monkeypatch):
    monkeypatch.setattr("pylab.show", lambda: None)
    filename = tmp_path / "test_plot.png"
    plot_3d_contour(
        simple_plot_data["X_combined"],
        simple_plot_data["Y_combined"],
        simple_plot_data["Z_combined"],
        simple_plot_data["min_z"],
        simple_plot_data["max_z"],
        filename=str(filename),
        show=False
    )
    assert os.path.exists(filename)

def test_plot_3d_contour_with_custom_figsize(simple_plot_data, monkeypatch):
    monkeypatch.setattr("pylab.show", lambda: None)
    plot_3d_contour(
        simple_plot_data["X_combined"],
        simple_plot_data["Y_combined"],
        simple_plot_data["Z_combined"],
        simple_plot_data["min_z"],
        simple_plot_data["max_z"],
        figsize=(8, 4),
        show=True
    )

def test_plot_3d_contour_with_custom_cmap(simple_plot_data, monkeypatch):
    monkeypatch.setattr("pylab.show", lambda: None)
    plot_3d_contour(
        simple_plot_data["X_combined"],
        simple_plot_data["Y_combined"],
        simple_plot_data["Z_combined"],
        simple_plot_data["min_z"],
        simple_plot_data["max_z"],
        cmap="viridis",
        show=True
    )

def test_plot_3d_contour_with_tkagg(simple_plot_data, monkeypatch):
    monkeypatch.setattr("pylab.show", lambda: None)
    # TkAgg backend cannot be loaded if another interactive backend is running
    with pytest.raises(ImportError):
        plot_3d_contour(
            simple_plot_data["X_combined"],
            simple_plot_data["Y_combined"],
            simple_plot_data["Z_combined"],
            simple_plot_data["min_z"],
            simple_plot_data["max_z"],
            tkagg=True,
            show=True
        )