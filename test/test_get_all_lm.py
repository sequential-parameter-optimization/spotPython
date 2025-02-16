import pytest
import pandas as pd
from spotpython.utils.stats import fit_all_lm

def test_fit_all_lm():
    # Test case 1: Basic model with one independent variable
    data = pd.DataFrame({
        'y': [1, 2, 3],
        'x1': [4, 5, 6],
        'x2': [7, 8, 9]
    })
    result = fit_all_lm("y ~ x1", ["x2"], data)
    expected_vars = ['basic', 'x2']
    assert list(result['estimate']['variables']) == expected_vars
    assert result['fun'] == 'all_lm'
    assert result['basic'] == 'y ~ x1'
    assert result['family'] == 'lm'

    # Test case 2: Model with multiple independent variables
    data = pd.DataFrame({
        'y': [1, 2, 3, 4],
        'x1': [4, 5, 6, 7],
        'x2': [7, 8, 9, 10],
        'x3': [10, 11, 12, 13]
    })
    result = fit_all_lm("y ~ x1", ["x2", "x3"], data)
    expected_vars = ['basic', 'x2', 'x3', 'x2, x3']
    assert list(result['estimate']['variables']) == expected_vars
    assert result['fun'] == 'all_lm'
    assert result['basic'] == 'y ~ x1'
    assert result['family'] == 'lm'

    # Test case 3: Model with missing values
    data = pd.DataFrame({
        'y': [1, 2, None, 4],
        'x1': [4, 5, 6, 7],
        'x2': [7, 8, 9, 10]
    })
    result = fit_all_lm("y ~ x1", ["x2"], data, remove_na=True)
    expected_vars = ['basic', 'x2']
    assert list(result['estimate']['variables']) == expected_vars
    assert result['fun'] == 'all_lm'
    assert result['basic'] == 'y ~ x1'
    assert result['family'] == 'lm'
    assert result['estimate']['n'].iloc[0] == 3  # Check if missing values were removed

if __name__ == "__main__":
    pytest.main()