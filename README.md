# mlinterp

This package is a Python wrapper to the C++ [mlinterp](https://github.com/parsiad/mlinterp) multidimensional linear interpolation library. The interface is designed to be very similar to the Scipy [RegularGridInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html), when initialized with `method='linear'`, `bounds_error=False`, and `fill_value=nan`. The benefit of the `mlinterp` version of `RegularGridInterpolator` is that it is a factor of >100x faster than Scipy when interpolating one point at a time.

## Usage

The usage is very similar the Scipy [RegularGridInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html):

```python
from mlinterp import RegularGridInterpolator
import numpy as np

def f(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
points = (x, y, z)
values = f(xg, yg, zg)
 
interp = RegularGridInterpolator(points, values)

xi = np.array([2.1, 6.2, 8.3])
print('%.1f'%interp(xi)) # interpolated value
print('%.1f'%f(2.1, 6.2, 8.3)) # function value
```

The result is

```
125.8
125.5
```

