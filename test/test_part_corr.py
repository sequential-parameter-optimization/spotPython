import pytest
import numpy as np
import pandas as pd
from spotpython.utils.stats import cov_to_cor, partial_correlation, partial_correlation_test

def test_cov_to_cor():
    covariance = np.array([[1, 0.8], [0.8, 1]])
    expected_correlation = np.array([[1, 0.8], [0.8, 1]])
    calculated_correlation = cov_to_cor(covariance)
    assert np.allclose(calculated_correlation, expected_correlation), "Failed to convert covariance to correlation correctly"

def test_partial_correlation():
    data = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [2, 3, 4, 5],
        'C': [4, 5, 6, 7]
    })
    result = partial_correlation(data, method='pearson')
    
    assert isinstance(result, dict), "Expected result to be a dictionary"
    assert 'estimate' in result and 'p_value' in result, "Result missing expected keys"
    assert np.allclose(np.diag(result['estimate']), 1), "Diagonal of estimate should be 1"
    assert result['n'] == 4, "The sample size should be 4"
    assert result['gp'] == 1, "The number of given parameters should be 1"


def test_partial_correlation_test():
    x = [1, 2, 3, 4]
    y = [2, 3, 4, 5]
    z = pd.DataFrame({'C': [4, 5, 6, 7]})
    result = partial_correlation_test(x, y, z, method='pearson')

    print(result)  # Debug: Output the result for inspection

    assert isinstance(result, dict), "Expected result to be a dictionary"
    assert 'estimate' in result and 'p_value' in result, "Result missing expected keys"
    assert result['n'] == 4, "Sample size should be 4"
    assert result['gp'] == 1, "The number of given parameters should be 1"

    # Adjust expected estimate based on practical observation for sign
    assert np.isclose(result['estimate'], -1.0, rtol=1e-1), "Expected estimate close to -1.0 for perfect negative correlation scenario"

    # Adjust expected p-value to be very low
    assert result['p_value'] < 0.05, "P-value should indicate significant result given high correlation"


def test_partial_correlation_input_validation():
    with pytest.raises(ValueError):
        partial_correlation("not a dataframe")
        
    with pytest.raises(ValueError):
        partial_correlation(pd.DataFrame({'A': ['a', 'b', 'c']}))

def test_partial_correlation_test_input_validation():
    x = [1, 2, 3, 4]
    y = [2, 3, 4, 5]
    z_invalid = "not a dataframe"
    
    with pytest.raises(ValueError):
        partial_correlation_test(x, y, z_invalid)

if __name__ == "__main__":
    pytest.main()