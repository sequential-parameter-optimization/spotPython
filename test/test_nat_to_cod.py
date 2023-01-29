def test_nat_to_cod():
    """
    Test nat_to_cod and cod_to_nat
    """
from spotPython.fun.objectivefunctions import analytical
from spotPython.design.factorial import factorial
from spotPython.build.kriging import Kriging
import numpy as np

gen = factorial(3)
rng = np.random.RandomState(1)
lower = np.array([-1,-1])
upper = np.array([0,0])
fun = analytical().fun_linear
X = gen.full_factorial(3)
X = 10*X
y = fun(X)

S = Kriging(name='kriging',  seed=123)
S.fit(X,y)
X2 = 2*X

Y = np.empty_like(X2)
T = np.empty_like(X2)
for i in range(S.n):
    T[i] = S.nat_to_cod_x(X2[i])
    Y[i] = S.cod_to_nat_x(T[i])

assert np.array_equal(Y, X2)
