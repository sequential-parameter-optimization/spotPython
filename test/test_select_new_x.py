import numpy as np
import pytest
from spotpython.utils.compare import selectNew

def test_selectNew_all_known():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    X = np.array([[1, 2, 3], [4, 5, 6]])
    B, ind = selectNew(A, X)
    assert B.shape[0] == 0
    assert np.all(ind == np.array([False, False]))

def test_selectNew_one_new():
    A = np.array([[1, 2, 3], [4, 5, 7]])
    X = np.array([[1, 2, 3], [4, 5, 6]])
    B, ind = selectNew(A, X)
    assert B.shape[0] == 1
    assert np.all(ind == np.array([False, True]))
    assert np.all(B == np.array([[4, 5, 7]]))

def test_selectNew_all_new():
    A = np.array([[7, 8, 9], [10, 11, 12]])
    X = np.array([[1, 2, 3], [4, 5, 6]])
    B, ind = selectNew(A, X)
    assert B.shape[0] == 2
    assert np.all(ind == np.array([True, True]))
    assert np.all(B == A)

def test_selectNew_with_tolerance():
    A = np.array([[1.0, 2.0, 3.0], [4.01, 5.0, 6.0]])
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B, ind = selectNew(A, X, tolerance=0.05)
    assert B.shape[0] == 0
    assert np.all(ind == np.array([False, False]))

def test_selectNew_with_tolerance_one_new():
    A = np.array([[1.0, 2.0, 3.0], [4.1, 5.0, 6.0]])
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B, ind = selectNew(A, X, tolerance=0.05)
    assert B.shape[0] == 1
    assert np.all(ind == np.array([False, True]))
    assert np.allclose(B, np.array([[4.1, 5.0, 6.0]]))