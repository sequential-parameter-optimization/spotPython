import pandas as pd
import pytest
from spotpython.utils.compare import check_identical_columns_and_rows, check_identical_columns_and_rows_with_tol

def test_check_exact_identical_columns_and_rows_remove_true():
    # Test DataFrame with exact duplicate columns
    df1 = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, 2, 3],
        "C": [4, 5, 6]
    })
    
    result_df, identical_cols, identical_rows = check_identical_columns_and_rows(df1, remove=True)
    assert list(result_df.columns) == ["A", "C"], "Failed to remove duplicate columns accurately"
    assert identical_cols == [("A", "B")], "Failed to identify exact duplicate columns"
    assert identical_rows == [], "Incorrectly identified duplicate rows where none exist"

def test_check_exact_identical_columns_and_rows_remove_false():
    # Test DataFrame checks presence of duplicates without removing
    df1 = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, 2, 3],
        "C": [4, 5, 6]
    })
    
    result_df, identical_cols, identical_rows = check_identical_columns_and_rows(df1, remove=False)
    assert list(result_df.columns) == ["A", "B", "C"], "Incorrectly identified or removed columns when remove=False"
    assert identical_cols == [("A", "B")], "Failed to identify exact duplicate columns"
    assert identical_rows == [], "Incorrectly found duplicate rows"

def test_check_identical_columns_and_rows_with_tol_remove_true():
    # Test DataFrame with near-duplicate tolerance checks for columns
    df1 = pd.DataFrame({
        "A": [1.00, 2.01, 3.00],
        "B": [1.01, 2.00, 3.01],
        "C": [4.00, 5.00, 6.00]
    })

    result_df, identical_cols, identical_rows = check_identical_columns_and_rows_with_tol(df1, tolerance=0.05, remove=True)
    assert list(result_df.columns) == ["A", "C"], "Failed to remove near-duplicate columns accurately with tolerance"
    assert identical_cols == [("A", "B")], "Failed to identify near-duplicate columns within tolerance"
    assert identical_rows == [], "Incorrectly found duplicate rows"

def test_check_identical_columns_and_rows_with_tol_remove_false():
    # Test DataFrame with tolerance execution, no removal
    df1 = pd.DataFrame({
        "A": [1.00, 2.01, 3.00],
        "B": [1.01, 2.00, 3.01],
        "C": [4.00, 5.00, 6.00]
    })

    result_df, identical_cols, identical_rows = check_identical_columns_and_rows_with_tol(df1, tolerance=0.05, remove=False)
    assert list(result_df.columns) == ["A", "B", "C"], "Incorrectly identified or removed columns when remove=False"
    assert identical_cols == [("A", "B")], "Failed to identify near-duplicate columns within tolerance"
    assert identical_rows == [], "Incorrectly found duplicate rows"

def test_with_no_duplicates():
    # Tests scenarios where there are no duplicate columns or rows present
    df = pd.DataFrame({
        "X": [1, 2, 3],
        "Y": [4, 5, 6],
        "Z": [7, 8, 9]
    })
    result_df, identical_cols, identical_rows = check_identical_columns_and_rows(df, remove=True)
    assert list(result_df.columns) == ["X", "Y", "Z"], "Incorrectly removed columns in a no-duplicates scenario"
    assert identical_cols == [], "Incorrectly found duplicate columns where none exist"
    assert identical_rows == [], "Incorrectly found duplicate rows where none exist"

    result_df_with_tol, identical_cols_with_tol, identical_rows_with_tol = check_identical_columns_and_rows_with_tol(df, tolerance=0.1, remove=True)
    assert list(result_df_with_tol.columns) == ["X", "Y", "Z"], "Incorrectly removed columns in a no-duplicates scenario with tolerance"
    assert identical_cols_with_tol == [], "Incorrectly found duplicate columns where none exist"
    assert identical_rows_with_tol == [], "Incorrectly found duplicate rows where none exist"

if __name__ == "__main__":
    pytest.main()