import copy
import numpy as np
from spotpython.fun.multiobjectivefunctions import MultiAnalytical
from spotpython.spot import Spot
from spotpython.utils.init import fun_control_init, surrogate_control_init, design_control_init


def test_repeats_1_1_0():
    """
    Test repeats. 1 repeat of initial design points, 1 repeat of function evaluations, and 0 repeats of OCBA.

    """

fun = MultiAnalytical(m=4).fun_mo_linear
surrogate_control = surrogate_control_init(method="regression")
design_control = design_control_init(init_size=3, repeats=1)
fun_control = fun_control_init(
    lower=np.array([-1]),
    upper=np.array([1]),
    fun_evals=20,
    fun_repeats=1,
    noise=True,
    ocba_delta=0,
    seed=123,
    show_models=False,
    show_progress=True,
    sigma=0.001,
)
S = Spot(
    fun=fun,
    surrogate_control=surrogate_control,
    design_control=design_control,
    fun_control=fun_control,
)
S.run()

def test_repeats_1_1_1():
    """
    Test repeats. 1 repeat of initial design points, 1 repeat of function evaluations, and 1 repeat of OCBA, i.e., ocba_delta=1
    """

fun = MultiAnalytical(m=2).fun_mo_linear
surrogate_control = surrogate_control_init(method="regression")
design_control = design_control_init(init_size=3, repeats=1)
fun_control = fun_control_init(
    lower=np.array([-1]),
    upper=np.array([1]),
    fun_evals=20,
    fun_repeats=1,
    noise=True,
    ocba_delta=1,
    seed=123,
    show_models=False,
    show_progress=True,
    sigma=0.001,
)
S = Spot(
    fun=fun,
    surrogate_control=surrogate_control,
    design_control=design_control,
    fun_control=fun_control,
)
S.run()

def test_repeats_2_1_0():
    """
    Test repeats. 2 repeat of initial design points, 1 repeat of function evaluations, and 1 repeat of OCBA, i.e., ocba_delta=0
    """

fun = MultiAnalytical(m=2).fun_mo_linear
surrogate_control = surrogate_control_init(method="regression")
design_control = design_control_init(init_size=3, repeats=2)
fun_control = fun_control_init(
    lower=np.array([-1]),
    upper=np.array([1]),
    fun_evals=20,
    fun_repeats=1,
    noise=True,
    ocba_delta=0,
    seed=123,
    show_models=False,
    show_progress=True,
    sigma=0.001,
)
S = Spot(
    fun=fun,
    surrogate_control=surrogate_control,
    design_control=design_control,
    fun_control=fun_control,
)
S.run()


def test_repeats_1_2_0():
    """
    Test repeats. 1 repeat of initial design points, 2 repeat of function evaluations, and 1 repeat of OCBA, i.e., ocba_delta=0
    """

fun = MultiAnalytical(m=2).fun_mo_linear
surrogate_control = surrogate_control_init(method="regression")
design_control = design_control_init(init_size=3, repeats=1)
fun_control = fun_control_init(
    lower=np.array([-1]),
    upper=np.array([1]),
    fun_evals=20,
    fun_repeats=2,
    noise=True,
    ocba_delta=0,
    seed=123,
    show_models=False,
    show_progress=True,
    sigma=0.001,
)
S = Spot(
    fun=fun,
    surrogate_control=surrogate_control,
    design_control=design_control,
    fun_control=fun_control,
)
S.run()


def test_repeats_1_2_1():
    """
    Test repeats. 1 repeat of initial design points, 2 repeat of function evaluations, and 1 repeat of OCBA, i.e., ocba_delta=1
    """

fun = MultiAnalytical(m=2).fun_mo_linear
surrogate_control = surrogate_control_init(method="regression")
design_control = design_control_init(init_size=3, repeats=1)
fun_control = fun_control_init(
    lower=np.array([-1]),
    upper=np.array([1]),
    fun_evals=20,
    fun_repeats=2,
    noise=True,
    ocba_delta=1,
    seed=123,
    show_models=False,
    show_progress=True,
    sigma=0.001,
)
S = Spot(
    fun=fun,
    surrogate_control=surrogate_control,
    design_control=design_control,
    fun_control=fun_control,
)
S.run()



def test_repeats_2_2_1():
    """
    Test repeats. 2 repeat of initial design points, 2 repeat of function evaluations, and 1 repeat of OCBA, i.e., ocba_delta=1
    """

fun = MultiAnalytical(m=2).fun_mo_linear
surrogate_control = surrogate_control_init(method="regression")
design_control = design_control_init(init_size=3, repeats=2)
fun_control = fun_control_init(
    lower=np.array([-1]),
    upper=np.array([1]),
    fun_evals=20,
    fun_repeats=2,
    noise=True,
    ocba_delta=1,
    seed=123,
    show_models=False,
    show_progress=True,
    sigma=0.001,
)
S = Spot(
    fun=fun,
    surrogate_control=surrogate_control,
    design_control=design_control,
    fun_control=fun_control,
)
S.run()

def test_repeats_2_2_2():
    """
    Test repeats. 2 repeat of initial design points, 2 repeat of function evaluations, and 1 repeat of OCBA, i.e., ocba_delta=2
    """

fun = MultiAnalytical(m=2).fun_mo_linear
surrogate_control = surrogate_control_init(method="regression")
design_control = design_control_init(init_size=3, repeats=2)
fun_control = fun_control_init(
    lower=np.array([-1]),
    upper=np.array([1]),
    fun_evals=20,
    fun_repeats=2,
    noise=True,
    ocba_delta=2,
    seed=123,
    show_models=False,
    show_progress=True,
    sigma=0.001,
)
S = Spot(
    fun=fun,
    surrogate_control=surrogate_control,
    design_control=design_control,
    fun_control=fun_control,
)
S.run()