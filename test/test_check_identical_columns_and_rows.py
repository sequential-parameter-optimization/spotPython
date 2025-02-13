import pandas as pd
import pytest
from spotpython.utils.compare import check_identical_columns_and_rows, check_identical_columns_and_rows_with_tol

def test_check_exact_identical_columns_and_rows():
    # Test DataFrames
    df1 = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, 2, 3],
        "C": [4, 5, 6]
    })
    
    df2 = pd.DataFrame({
        "X": [7, 8, 9],
        "Y": [10, 11, 12]
    })

    # Exact duplicates - should identify and remove B
    result_df = check_identical_columns_and_rows(df1, remove=True)
    assert list(result_df.columns) == ["A", "C"], "Failed to remove duplicate columns accurately"

    # No duplicates - should not remove any columns
    result_df = check_identical_columns_and_rows(df2, remove=True)
    assert list(result_df.columns) == ["X", "Y"], "Incorrectly removed columns when there were none to remove"

def test_check_identical_columns_and_rows_with_tol():
    # Test DataFrame
    df1 = pd.DataFrame({
        "A": [1.00, 2.01, 3.00],
        "B": [1.01, 2.00, 3.01],
        "C": [4.00, 5.00, 6.00]
    })

    # Within-tolerance duplicates - should identify and remove B
    result_df = check_identical_columns_and_rows_with_tol(df1, tolerance=0.05, remove=True)
    assert list(result_df.columns) == ["A", "C"], "Failed to remove near-duplicate columns accurately"

    # No near duplicates within a small tolerance
    result_df = check_identical_columns_and_rows_with_tol(df1, tolerance=0.001, remove=True)
    assert list(result_df.columns) == ["A", "B", "C"], "Incorrectly removed columns when they are not near duplicates"

def test_check_exact_identical_columns_and_rows_remove_true():
    df1 = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, 2, 3],
        "C": [4, 5, 6]
    })
    
    result_df = check_identical_columns_and_rows(df1, remove=True)
    assert list(result_df.columns) == ["A", "C"], "Failed to remove duplicate columns accurately"

def test_check_exact_identical_columns_and_rows_remove_false():
    df1 = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, 2, 3],
        "C": [4, 5, 6]
    })
    
    # Check without removing duplicates
    result_df = check_identical_columns_and_rows(df1, remove=False)
    assert list(result_df.columns) == ["A", "B", "C"], "Incorrectly identified or removed columns when remove=False"

def test_check_identical_columns_and_rows_with_tol_remove_true():
    df1 = pd.DataFrame({
        "A": [1.00, 2.01, 3.00],
        "B": [1.01, 2.00, 3.01],
        "C": [4.00, 5.00, 6.00]
    })

    result_df = check_identical_columns_and_rows_with_tol(df1, tolerance=0.05, remove=True)
    assert list(result_df.columns) == ["A", "C"], "Failed to remove near-duplicate columns accurately with tolerance"

def test_check_identical_columns_and_rows_with_tol_remove_false():
    df1 = pd.DataFrame({
        "A": [1.00, 2.01, 3.00],
        "B": [1.01, 2.00, 3.01],
        "C": [4.00, 5.00, 6.00]
    })

    # Check without removing duplicates
    result_df = check_identical_columns_and_rows_with_tol(df1, tolerance=0.05, remove=False)
    assert list(result_df.columns) == ["A", "B", "C"], "Incorrectly identified or removed columns when remove=False"

def test_with_no_duplicates():
    df = pd.DataFrame({
        "X": [1, 2, 3],
        "Y": [4, 5, 6],
        "Z": [7, 8, 9]
    })
    result_df = check_identical_columns_and_rows(df, remove=True)
    assert list(result_df.columns) == ["X", "Y", "Z"], "Incorrectly removed columns in a no-duplicates scenario"

    result_df_with_tol = check_identical_columns_and_rows_with_tol(df, tolerance=0.1, remove=True)
    assert list(result_df_with_tol.columns) == ["X", "Y", "Z"], "Incorrectly removed columns in a no-duplicates scenario with tolerance"


if __name__ == "__main__":
    pytest.main()
