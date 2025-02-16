import pytest
from spotpython.utils.stats import get_all_vars_from_formula

def test_get_all_vars_from_formula():
    # Test case 1: Simple formula
    formula = "y ~ x1 + x2"
    expected_vars = ['y', 'x1', 'x2']
    assert get_all_vars_from_formula(formula) == expected_vars

    # Test case 2: Formula with extra spaces
    formula = "  y  ~  x1  +  x2  "
    expected_vars = ['y', 'x1', 'x2']
    assert get_all_vars_from_formula(formula) == expected_vars

    # Test case 3: Formula with multiple independent variables
    formula = "y ~ x1 + x2 + x3 + x4"
    expected_vars = ['y', 'x1', 'x2', 'x3', 'x4']
    assert get_all_vars_from_formula(formula) == expected_vars

    # Test case 4: Formula with no independent variables
    formula = "y ~ "
    expected_vars = ['y']
    assert get_all_vars_from_formula(formula) == expected_vars

    # Test case 5: Formula with only one independent variable
    formula = "y ~ x1"
    expected_vars = ['y', 'x1']
    assert get_all_vars_from_formula(formula) == expected_vars

if __name__ == "__main__":
    pytest.main()