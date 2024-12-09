# mlinterpy

This package is a Python wrapper to the C++ [mlinterp](https://github.com/parsiad/mlinterp) library, which does multidimensional linear interpolation on regular grids. The interface is designed to be very similar to the Scipy [RegularGridInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html), when initialized with `method='linear'` and `bounds_error=False`. The benefit of the `mlinterpy` is that it is generally much faster than Scipy. Also, `mlinterp` contains a version of the its `MultiLinearInterpolator` that can be used within [numba](https://numba.pydata.org/) compiled functions.

## Usage

The usage is very similar the Scipy [RegularGridInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html):

```python
from mlinterpy import MultiLinearInterpolator
import numpy as np

def f(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
points = (x, y, z)
values = f(xg, yg, zg)
 
interp = MultiLinearInterpolator(points, values)

xi = np.array([2.1, 6.2, 8.3])
print('%.1f'%interp(xi)[0]) # interpolated value
print('%.1f'%interp.evaluate(xi)) # interpolated value (faster than previous line)
print('%.1f'%f(2.1, 6.2, 8.3)) # function value
```

The result is

```
125.8
125.8
125.5
```

The code below compares with the scipy `RegularGridInterpolator`:

```python
from scipy import interpolate
import timeit

interp_scipy = interpolate.RegularGridInterpolator(points, values)
def test_scipy():
    return interp_scipy(xi)[0]

def test():
    return interp.evaluate(xi)

assert np.isclose(test_scipy(),test())

timer = timeit.Timer(test_scipy)
n, _ = timer.autorange()
t_scipy = timer.timeit(number=n)/n

timer = timeit.Timer(test)
n, _ = timer.autorange()
t = timer.timeit(number=n)/n

print('mlinterpy is %i times faster than scipy'%(t_scipy/t))
```

The result: `mlinterpy is 512 times faster than scipy`

