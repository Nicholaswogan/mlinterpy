
import numpy as np
from mlinterpy import MultiLinearInterpolator
from scipy import interpolate
import timeit

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def timefunc(func):
    timer = timeit.Timer(func)
    number, t = timer.autorange()
    return t/number

def test_numpy():
    
    x = np.linspace(0,10,10)
    y = np.sin(x)

    xi = np.linspace(0,10,1000)
    tmp = lambda: np.interp(xi, x, y)

    res1 = tmp()
    t1 = timefunc(tmp)

    interp = MultiLinearInterpolator((x,), y)
    xi1 = np.ascontiguousarray(xi.reshape(1,xi.shape[0]))
    tmp = lambda: interp.evaluate_vector(xi1)

    res2 = tmp()
    t2 = timefunc(tmp)

    print('numpy time: %.1e s, mlinterpy time: %.1e s, numpy/mlinterpy: %.3f'%(t1, t2, t1/t2))
    assert np.allclose(res1, res2, atol=1e-100, rtol=1e-10)

def test_interp2d():

    # --------- interp2d
    x = np.linspace(0, 4, 13)
    y = np.array([0, 2, 3, 3.5, 3.75, 3.875, 3.9375, 4])
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.pi * X / 2) * np.exp(Y / 2)

    x2 = np.linspace(0, 4, 65)
    y2 = np.linspace(0, 4, 65)
    f = interpolate.interp2d(x, y, Z, kind="linear")

    tmp = lambda: f(x2, y2)
    Z2 = tmp()
    t1 = timefunc(tmp)

    # --------- RegularGridInterpolator
    X, Y = np.meshgrid(x, y, indexing="ij")  # indexing="ij" is important
    Z = np.sin(np.pi * X / 2) * np.exp(Y / 2)
    f2 = MultiLinearInterpolator((x, y), Z)

    X2, Y2 = np.meshgrid(x2, y2)
    xi = np.array([X2.flatten(), Y2.flatten()])
    xi = np.ascontiguousarray(xi)

    tmp = lambda: f2.evaluate_vector(xi)    
    Z3 = tmp().reshape(X2.shape)
    t2 = timefunc(tmp)
    
    print('interp2d time: %.1e s, mlinterpy time: %.1e s, interp2d/mlinterpy: %.3f'%(t1, t2, t1/t2))

    assert np.allclose(Z2, Z3)

if __name__ == '__main__':
    test_numpy()
    test_interp2d()







