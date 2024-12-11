from . import _mlinterpy
import numba as nb
from numba import types
import numpy as np
import ctypes as ct

def make_interp_wrapper():
    lib = ct.CDLL(_mlinterpy.__file__)
    interp_wrapper = lib.interp_wrapper
    interp_wrapper.argtypes = [
        ct.c_int,    # int ndim
        ct.c_void_p, # int *nd
        ct.c_void_p, # double **xd
        ct.c_void_p, # double *fd
        ct.c_int,    # int ni
        ct.c_void_p, # double *xi
        ct.c_void_p  # double *fi
    ]
    interp_wrapper.restype = ct.c_int
    return interp_wrapper

interp_wrapper = make_interp_wrapper()

@nb.njit(types.bool_(types.Array(types.double, 1, 'C', readonly=True)))
def is_sorted(a):
  for i in range(a.shape[0] - 1):
      if not a[i+1] > a[i]:
          return False
  return True

@nb.experimental.jitclass()
class MultiLinearInterpolator():

    ndim: types.intc # type: ignore
    nd: types.Array(types.intc, 1, 'C') # type: ignore
    points: types.List(types.Array(types.double, 1, 'C'), True) # type: ignore
    xd: types.Array(types.intp, 1, 'C') # type: ignore
    fd: types.Array(types.double, 1, 'C') # type: ignore
    
    def __init__(self, points, values):
        self.ndim = len(points)
        self.nd = np.empty(self.ndim,np.intc)
        self.xd = np.empty(self.ndim,np.intp)

        points_list = []
        for i in range(self.ndim):
            points_list.append(points[i])
        self.points = points_list

        for i in range(self.ndim):
            self.nd[i] = len(self.points[i])
            assert self.nd[i] > 0, "All arrays in input `points` must be longer than 0."
            assert self.nd[i] == values.shape[i], "Input `points` and `values` have incompatible shapes"
            self.xd[i] = self.points[i].ctypes.data
            sort = is_sorted(self.points[i])
            if not sort:
                raise ValueError('Some of the arrays in `points` are not sorted')

        values_view = values.ravel()
        self.fd = np.empty(values.size,np.double)
        for i in range(values.size):
            self.fd[i] = values_view[i]

    def evaluate_vector(self, xi):
        return _evaluate_vector(self, xi)

    def evaluate(self, xi):
        return _evaluate(self, xi)
    
    def call(self, xi):

        if xi.ndim == 1:
            tmp = self.evaluate(xi)
            fi = np.array([tmp])
        elif xi.ndim == 2:
            fi = self.evaluate_vector(xi)
        else:
            raise ValueError("`xi` must have 1 or 2 dimensions.")

        return fi

_signature = types.Array(types.double, 1, 'C')(
    MultiLinearInterpolator.class_type.instance_type,
    types.Array(types.double, 2, 'C', readonly=True),
)
@nb.njit(_signature)
def _evaluate_vector(self, xi):

    ni = xi.shape[1]
    assert xi.shape[0] == self.ndim, "Input `xi` has the wrong dimension"
    fi = np.empty(ni, np.double)
    
    ierr = interp_wrapper(
        self.ndim,
        self.nd.ctypes.data,
        self.xd.ctypes.data,
        self.fd.ctypes.data,
        ni,
        xi.ctypes.data,
        fi.ctypes.data
    )
    if ierr:
        raise Exception('Memory allocation failed.')
    
    return fi

_signature = types.double(
    MultiLinearInterpolator.class_type.instance_type,
    types.Array(types.double, 1, 'C', readonly=True)
)
@nb.njit(_signature)
def _evaluate(self, xi):

    assert xi.shape[0] == self.ndim, "Input `xi` has the wrong dimension"
    fi = np.array(0.0,np.double)
    
    ierr = interp_wrapper(
        self.ndim,
        self.nd.ctypes.data,
        self.xd.ctypes.data,
        self.fd.ctypes.data,
        1,
        xi.ctypes.data,
        fi.ctypes.data
    )
    if ierr:
        raise Exception('Memory allocation failed.')
    
    return fi.item()
