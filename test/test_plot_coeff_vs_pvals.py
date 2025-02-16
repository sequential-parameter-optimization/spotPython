import pytest
import pandas as pd
from spotpython.utils.stats import fit_all_lm, plot_coeff_vs_pvals
import matplotlib.pyplot as plt

def test_plot_coeff_vs_pvals():
    # Test case: Basic functionality
    data = pd.DataFrame({
        'y': [1, 2, 3],
        'x1': [4, 5, 6],
        'x2': [7, 8, 9]
    })
    estimates = fit_all_lm("y ~ x1", ["x2"], data)

    # Test plotting without showing the plot
    plot_coeff_vs_pvals(estimates, show=False)

    # Test plotting with custom xlabels and title
    plot_coeff_vs_pvals(estimates, xlabels=[0, 0.01, 0.05, 0.1, 0.5, 1], title="Test Plot", show=False)

    # Test plotting with log scale on both axes
    plot_coeff_vs_pvals(estimates, xscale_log=True, yscale_log=True, show=False)

    # Test plotting with custom xlim and ylim
    plot_coeff_vs_pvals(estimates, xlim=(0.001, 1), ylim=(-2, 2), show=False)

    # Test plotting with custom xlab and ylab
    plot_coeff_vs_pvals(estimates, xlab="Custom X Label", ylab="Custom Y Label", show=False)

    # Ensure no exceptions are raised during plotting
    assert True

if __name__ == "__main__":
    pytest.main()