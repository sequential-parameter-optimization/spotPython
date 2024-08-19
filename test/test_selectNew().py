from spotpython.utils.compare import selectNew
import numpy as np


def test_selectNew_All_Equal():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    X = np.array([[1, 2, 3], [4, 5, 6]])
    B, ind = selectNew(A, X)
    assert B.shape[0] == 0
    assert np.equal(ind, np.array([False, False])).all()


def test_selectNew_One_Not_Equal():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    X = np.array([[1, 2, 3], [4, 5, 7]])
    B, ind = selectNew(A, X)
    assert B.shape[0] == 1
    assert np.equal(ind, np.array([False, True])).all()
