import numpy as np
import numba as nb
from numba import types

@nb.njit()
def mux(ndim, nd, indices):
    index = 0
    product = 1
    i = ndim - 1
    while True:
        index += indices[i] * product
        if i == 0:
            break
        product *= nd[i]
        i -= 1
    return index

@nb.njit()
def run(ndim, nd, xd, n, ni, xi, indicies, weights):

    for i in range(ndim):
        x = xi[n + i*ni]
        xdi = xd[i]

        if nd[i] == 1 or x <= xdi[0]:
            # Data point is less than left boundary
            mid = 0
            weight = 1.0
        elif x >= xdi[nd[i]-1]:
            # Data point is greater than right boundary
            mid = nd[i] - 1
            weight = 0.0
        else:
            # Binary search to find tick
            lo = 0
            hi = nd[i] - 2
            mid = 0
            weight = 0.0
            while lo <= hi:
                mid = int(lo + (hi - lo) / 2)
                if x < xdi[mid]:
                    hi = mid - 1
                elif x >= xdi[mid + 1]:
                    lo = mid + 1
                else:
                    weight = (xdi[mid + 1] - x) / (xdi[mid + 1] - xdi[mid])
                    break
                
        indicies[i] = mid
        weights[i] = weight

@nb.njit()
def interp_array(ndim, nd, xd, yd, xi, yi, indicies, weights, buffer):

    power = 1 << ndim

    run(ndim, nd, xd, 0, 1, xi, indicies, weights)

    arr_size = len(yd)
    for n in range(arr_size):
        yi[n] = 0.0
        for bitstr in range(power):
            factor = 1.0
            for i in range(ndim):
                if bitstr & (1 << i):
                    buffer[i] = indicies[i]
                    factor *= weights[i]
                else:
                    buffer[i] = indicies[i] + 1
                    factor *= 1 - weights[i]    
            if factor > 0:
                k = mux(ndim, nd, buffer)
                yi[n] += factor*yd[n][k]

@nb.njit()
def interp(ndim, nd, xd, yd, ni, xi, yi, indicies, weights, buffer):

    power = 1 << ndim

    for n in range(ni):
        yi[n] = 0.0
        run(ndim, nd, xd, n, ni, xi, indicies, weights)
        for bitstr in range(power):
            factor = 1.0
            for i in range(ndim):
                if bitstr & (1 << i):
                    buffer[i] = indicies[i]
                    factor *= weights[i]
                else:
                    buffer[i] = indicies[i] + 1
                    factor *= 1 - weights[i]    
            if factor > 0:
                k  = mux(ndim, nd, buffer)
                yi[n] += factor*yd[k]

@nb.njit(types.bool_(types.Array(types.double, 1, 'C', readonly=True)))
def is_sorted(a):
  for i in range(a.shape[0] - 1):
      if not a[i+1] > a[i]:
          return False
  return True

def interp1(xi, points, values):

    ndim = len(points)
    nd = np.empty(ndim,np.intc)
    for i in range(ndim):
        nd[i] = len(points[i])
    yi = np.empty(len(values),np.double)
    indicies = np.empty(ndim,np.intc)
    weights = np.empty(ndim,np.double)
    buffer = np.empty(ndim,np.intc)

    interp_array(
        ndim,
        nd,
        points,
        values,
        xi,
        yi,
        indicies,
        weights,
        buffer
    )

    return yi

@nb.experimental.jitclass()
class ArrayInterpolator():
    
    ndim: types.intc # type: ignore
    nd: types.Array(types.intc, 1, 'C') # type: ignore
    xd: types.List(types.Array(types.double, 1, 'C')) # type: ignore
    yd: types.List(types.Array(types.double, 1, 'C')) # type: ignore
    indicies: types.Array(types.intc, 1, 'C') # type: ignore
    weights: types.Array(types.double, 1, 'C') # type: ignore
    buffer: types.Array(types.intc, 1, 'C') # type: ignore
    
    def __init__(self, points, values):
        self.ndim = len(points)
        self.nd = np.empty(self.ndim,np.intc)

        points_list = []
        for i in range(self.ndim):
            points_list.append(points[i])
        self.xd = points_list

        for i in range(self.ndim):
            self.nd[i] = len(self.xd[i])
            assert self.nd[i] > 0, "All arrays in input `points` must be longer than 0."
            if not is_sorted(self.xd[i]):
                raise ValueError('Some of the arrays in `points` are not sorted')

        values_list = []
        for i in range(len(values)):
            assert values[i].ndim == self.ndim, "Input `points` and `values` have incompatible shapes"
            for j in range(self.ndim):
                assert self.nd[j] == values[i].shape[j], "Input `points` and `values` have incompatible shapes"
            values_list.append(values[i].ravel())
        self.yd = values_list
        self.indicies = np.empty(self.ndim,np.intc)
        self.weights = np.empty(self.ndim,np.double)
        self.buffer = np.empty(self.ndim,np.intc)

    def evaluate(self, xi):
        return _evaluate1(self, xi)
    
_signature = types.Array(types.double, 1, 'C')(
    ArrayInterpolator.class_type.instance_type,
    types.Array(types.double, 1, 'C', readonly=True)
)
@nb.njit(_signature)
def _evaluate1(self, xi):

    assert xi.shape[0] == self.ndim, "Input `xi` has the wrong dimension"
    yi = np.empty(len(self.yd),np.double)

    interp_array(
        self.ndim,
        self.nd,
        self.xd,
        self.yd,
        xi,
        yi,
        self.indicies,
        self.weights,
        self.buffer
    )

    return yi

@nb.experimental.jitclass()
class MultiLinearInterpolator():

    ndim: types.intc # type: ignore
    nd: types.Array(types.intc, 1, 'C') # type: ignore
    xd: types.List(types.Array(types.double, 1, 'C')) # type: ignore
    yd: types.Array(types.double, 1, 'C') # type: ignore
    indicies: types.Array(types.intc, 1, 'C') # type: ignore
    weights: types.Array(types.double, 1, 'C') # type: ignore
    buffer: types.Array(types.intc, 1, 'C') # type: ignore
    
    def __init__(self, points, values):
        self.ndim = len(points)
        self.nd = np.empty(self.ndim,np.intc)

        points_list = []
        for i in range(self.ndim):
            points_list.append(points[i])
        self.xd = points_list

        assert values.ndim == self.ndim, "Input `points` and `values` have incompatible shapes"
        for i in range(self.ndim):
            self.nd[i] = len(self.xd[i])
            assert self.nd[i] > 0, "All arrays in input `points` must be longer than 0."
            assert self.nd[i] == values.shape[i], "Input `points` and `values` have incompatible shapes"
            if not is_sorted(self.xd[i]):
                raise ValueError('Some of the arrays in `points` are not sorted')

        self.yd = values.ravel()
        self.indicies = np.empty(self.ndim,np.intc)
        self.weights = np.empty(self.ndim,np.double)
        self.buffer = np.empty(self.ndim,np.intc)

    def evaluate_vector(self, xi):
        return _evaluate_vector(self, xi)

    def evaluate(self, xi):
        return _evaluate(self, xi)

_signature = types.Array(types.double, 1, 'C')(
    MultiLinearInterpolator.class_type.instance_type,
    types.Array(types.double, 2, 'C', readonly=True),
)
@nb.njit(_signature)
def _evaluate_vector(self, xi):

    ni = xi.shape[1]
    assert xi.shape[0] == self.ndim, "Input `xi` has the wrong dimension"
    yi = np.empty(ni, np.double)
    
    interp(
        self.ndim,
        self.nd,
        self.xd,
        self.yd,
        ni,
        xi.ravel(),
        yi,
        self.indicies,
        self.weights,
        self.buffer
    )

    return yi

_signature = types.double(
    MultiLinearInterpolator.class_type.instance_type,
    types.Array(types.double, 1, 'C', readonly=True)
)
@nb.njit(_signature)
def _evaluate(self, xi):

    assert xi.shape[0] == self.ndim, "Input `xi` has the wrong dimension"
    yi = np.array([0.0],np.double)
    
    interp(
        self.ndim,
        self.nd,
        self.xd,
        self.yd,
        1,
        xi,
        yi,
        self.indicies,
        self.weights,
        self.buffer
    )

    return yi[0]



