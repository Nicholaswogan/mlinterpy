from numpy cimport ndarray
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern int interp_wrapper(
  int n, int *nd, double **xd, double *fd,
  int ni, double *xi, double *fi
)
cdef extern int interp_array_wrapper(
  int ndim, const int *nd, double **xd, int arr_size, double **yd,
  const double *xi, double *yi
)

cdef is_sorted(int n, double *a):
  cdef int i
  for i in range(n-1):
      if not a[i+1] > a[i]:
          return False
  return True

cdef class ArrayInterpolator:
  """
  Linear interpolator on a regular grid for.

  Parameters
  ----------
  points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
      The points defining the regular grid in n dimensions. The points in
      each dimension (i.e. every elements of the points tuple) must be
      strictly ascending.

  values : list of ndarray of float
      The data on the regular grid in n dimensions.
  """

  cdef int ndim
  cdef int* nd
  cdef double** xd
  cdef int arr_size
  cdef double** fd

  def __cinit__(self, *args, **kwargs):
    self.xd = NULL
    self.nd = NULL
    self.fd = NULL

  def __init__(self, points, values):
    self.ndim = len(points)
    self.nd = <int *> malloc(self.ndim * sizeof(int))
    self.xd = <double **> malloc(self.ndim * sizeof(double*))

    cdef ndarray[double, ndim=1] tmp;
    cdef double *tmp_p;
    cdef int n1 = 1
    for i in range(self.ndim):
      tmp = points[i]
      tmp_p = <double *> tmp.data
      self.nd[i] = points[i].shape[0]
      n1 *= self.nd[i]
      assert self.nd[i] > 0, "All arrays in input `points` must be longer than 0."
      self.xd[i] = <double *> malloc(self.nd[i] * sizeof(double))
      for j in range(self.nd[i]):
        self.xd[i][j] = tmp_p[j]
      if not is_sorted(self.nd[i], self.xd[i]):
        raise ValueError('Some of the arrays in `points` are not sorted')

    cdef double *fd
    cdef ndarray values_copy
    self.arr_size = len(values)
    self.fd = <double **> malloc(self.arr_size * sizeof(double**))
    for j in range(self.arr_size):
      assert self.ndim == values[j].ndim, "Input `points` and `values` have incompatible shapes"
      for i in range(self.ndim):
        assert self.nd[i] == values[j].shape[i], "Input `points` and `values` have incompatible shapes"
      assert values[j].dtype == np.double, "`values` must have have dtype `np.double`"
    
      if np.PyArray_IS_C_CONTIGUOUS(values[j]):
        values_copy = values[j].view()
        fd = <double *> values_copy.data
      else:
        values_copy = np.ascontiguousarray(values[j])
        fd = <double *> values_copy.data
      self.fd[j] = <double *> malloc(n1 * sizeof(double))
      for i in range(n1):
        self.fd[j][i] = fd[i]
    
  def __dealloc__(self):
    if self.xd:
      for i in range(self.ndim):
        free(self.xd[i])
      free(self.xd)
    if self.nd: 
      free(self.nd)
    if self.fd:
      for i in range(self.arr_size):
        free(self.fd[i])
      free(self.fd)

  def evaluate(self, ndarray[double, ndim=1] xi):

    assert xi.shape[0] == self.ndim, "Input `xi` has the wrong dimension"
    cdef ndarray[double,ndim=1] fi = np.empty(self.arr_size, np.double)

    cdef double *xi_p
    cdef ndarray[double,ndim=1] xi_copy
    if np.PyArray_IS_C_CONTIGUOUS(xi):
      xi_p = <double *> xi.data
    else:
      xi_copy = np.ascontiguousarray(xi)
      xi_p = <double *> xi_copy.data

    cdef int ierr = interp_array_wrapper(
      self.ndim, self.nd, self.xd, self.arr_size, self.fd, 
      xi_p, <double *> fi.data
    )
    if ierr:
      raise Exception('Memory allocation failed.')
    
    return fi

cdef class MultiLinearInterpolator:
  """
  Linear interpolator on a regular grid.

  Parameters
  ----------
  points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
      The points defining the regular grid in n dimensions. The points in
      each dimension (i.e. every elements of the points tuple) must be
      strictly ascending.

  values : ndarray, shape (m1, ..., mn, ...)
      The data on the regular grid in n dimensions.
  """

  cdef int ndim
  cdef int* nd
  cdef double** xd
  cdef double* fd

  def __cinit__(self, *args, **kwargs):
    self.xd = NULL
    self.nd = NULL
    self.fd = NULL

  def __init__(self, points, ndarray values):
    self.ndim = len(points)
    self.nd = <int *> malloc(self.ndim * sizeof(int))
    self.xd = <double **> malloc(self.ndim * sizeof(double*))

    cdef sort
    cdef ndarray[double, ndim=1] tmp;
    cdef double *tmp_p;
    for i in range(self.ndim):
      tmp = points[i]
      tmp_p = <double *> tmp.data
      self.nd[i] = points[i].shape[0]
      assert self.nd[i] > 0, "All arrays in input `points` must be longer than 0."
      assert self.nd[i] == values.shape[i], "Input `points` and `values` have incompatible shapes"
      self.xd[i] = <double *> malloc(self.nd[i] * sizeof(double))
      for j in range(self.nd[i]):
        self.xd[i][j] = tmp_p[j]
      sort = is_sorted(self.nd[i], self.xd[i])
      if not sort:
        raise ValueError('Some of the arrays in `points` are not sorted')

    assert values.dtype == np.double, "`values` must have have dtype `np.double`"
    cdef int n1 = values.size
    cdef double *fd
    cdef ndarray values_copy
    if np.PyArray_IS_C_CONTIGUOUS(values):
      fd = <double *> values.data
    else:
      values_copy = np.ascontiguousarray(values)
      fd = <double *> values_copy.data
    self.fd = <double *> malloc(n1 * sizeof(double))
    for i in range(n1):
      self.fd[i] = fd[i]
    
  def __dealloc__(self):
    if self.xd:
      for i in range(self.ndim):
        free(self.xd[i])
      free(self.xd)
    if self.nd: 
      free(self.nd)
    if self.fd: 
      free(self.fd)

  def evaluate_vector(self, ndarray[double, ndim=2] xi):
    """
    Interpolate at a many set of values `xi`.

    Parameters
    ----------
    xi : ndarray[double, ndim=2]
        The coordinates to evaluate the interpolator at. `xi` should
        have shape `(ndim,ni)`, where `ndim` is the number of dimensions,
        and `ni` is the number of points to do interpolation at.

    Returns
    -------
    fi : ndarray[double, ndim=1]
        Interpolated values at `xi`. `fi` has shape `xi.shape[1]`.
    """

    cdef int ni = xi.shape[1]
    assert xi.shape[0] == self.ndim, "Input `xi` has the wrong dimension"
    cdef ndarray[double,ndim=1] fi = np.empty(ni,np.double)

    cdef double *xi_p
    cdef ndarray[double,ndim=2] xi_copy
    if np.PyArray_IS_C_CONTIGUOUS(xi):
      xi_p = <double *> xi.data
    else:
      xi_copy = np.ascontiguousarray(xi)
      xi_p = <double *> xi_copy.data

    cdef int ierr = interp_wrapper(
      self.ndim, self.nd, self.xd, self.fd, 
      ni, xi_p, <double *> fi.data
    )
    if ierr:
      raise Exception('Memory allocation failed.')

    return fi

  def evaluate(self, ndarray[double, ndim=1] xi):
    """
    Interpolate at a single set of values `xi`.

    Parameters
    ----------
    xi : ndarray[double, ndim=1]
        The coordinates to evaluate the interpolator at.

    Returns
    -------
    fi : float
        Interpolated value at `xi`.
    """

    assert xi.shape[0] == self.ndim, "Input `xi` has the wrong dimension"
    cdef double fi

    cdef double *xi_p
    cdef ndarray[double,ndim=1] xi_copy
    if np.PyArray_IS_C_CONTIGUOUS(xi):
      xi_p = <double *> xi.data
    else:
      xi_copy = np.ascontiguousarray(xi)
      xi_p = <double *> xi_copy.data

    cdef int ierr = interp_wrapper(
      self.ndim, self.nd, self.xd, self.fd, 
      1, xi_p, &fi
    )
    if ierr:
      raise Exception('Memory allocation failed.')
    
    return fi

  def call(self, ndarray xi):
    """
    Interpolation at coordinates.

    Parameters
    ----------
    xi : ndarray
        The coordinates to evaluate the interpolator at.

    Returns
    -------
    fi : ndarray[double, ndim=1]
        Interpolated values at `xi`.
    """

    cdef double tmp;
    cdef ndarray[double,ndim=1] fi

    if xi.ndim == 1:
      tmp = self.evaluate(xi)
      fi = np.array([tmp])
    elif xi.ndim == 2:
      fi = self.evaluate_vector(xi)
    else:
      raise ValueError("`xi` must have 1 or 2 dimensions.")

    return fi
  
  def __call__(self, ndarray xi):
    """
    Interpolation at coordinates.

    Parameters
    ----------
    xi : ndarray
        The coordinates to evaluate the interpolator at.

    Returns
    -------
    fi : ndarray[double, ndim=1]
        Interpolated values at `xi`.
    """
    return self.call(xi)
