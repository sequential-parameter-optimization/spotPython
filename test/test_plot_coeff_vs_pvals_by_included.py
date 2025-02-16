import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spotpython.utils.stats import plot_coeff_vs_pvals_by_included

def test_plot_coeff_vs_pvals_by_included():
    # Test case: Basic functionality
    data = {
        "estimate": pd.DataFrame({
            "variables": ["Crude", "AL", "AM", "AN", "AO"],
            "estimate": [0.5, 0.6, 0.7, 0.8, 0.9],
            "conf_low": [0.1, 0.2, 0.3, 0.4, 0.5],
            "conf_high": [0.9, 1.0, 1.1, 1.2, 1.3],
            "p": [0.01, 0.02, 0.03, 0.04, 0.05],
            "aic": [100, 200, 300, 400, 500],
            "n": [10, 20, 30, 40, 50]
        }),
        "xlist": ["AL", "AM", "AN", "AO"],
        "fun": "all_lm"
    }

    # Test plotting without showing the plot
    plot_coeff_vs_pvals_by_included(data, show=False)

    # Test plotting with custom xlabels and title
    plot_coeff_vs_pvals_by_included(data, xlabels=[0, 0.01, 0.05, 0.1, 0.5, 1], title="Test Plot", show=False)

    # Test plotting with log scale on y-axis
    plot_coeff_vs_pvals_by_included(data, yscale_log=True, show=False)

    # Test plotting with custom xlim and ylim
    plot_coeff_vs_pvals_by_included(data, xlim=(0.001, 1), ylim=(-2, 2), show=False)

    # Test plotting with custom xlab and ylab
    plot_coeff_vs_pvals_by_included(data, xlab="Custom X Label", ylab="Custom Y Label", show=False)

    # Ensure no exceptions are raised during plotting
    assert True

if __name__ == "__main__":
    pytest.main()