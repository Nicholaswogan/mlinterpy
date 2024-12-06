#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "mlinterp.hpp"
using namespace mlinterp;

inline int mux(int dimension, const int *nd, const int *indices) {
    int index = 0, product = 1, i = dimension - 1;
    while (1) {
      index += indices[i] * product;
      if (i == 0) {
        break;
      }
      product *= nd[i--];
    }
    return index;
}

inline void run(
  int ndim, const int *nd, double **xd, 
  int n, const int ni, const double *xi,
  int *indices, double *weights)
{

  double *xdi;
  double x, weight;
  int mid;

  for (int i = 0; i < ndim; i++){
    x = xi[n + i*ni];
    xdi = xd[i];
    if (nd[i] == 1 || x <= xdi[0]) {
      // Data point is less than left boundary
      mid = 0;
      weight = 1.0;
    } else if (x >= xdi[nd[i]-1]) {
      // Data point is greater than right boundary
      mid = nd[i] - 1;
      weight = 0.0;
    } else {
      // Binary search to find tick
      int lo = 0, hi = nd[i] - 2;
      mid = 0;
      weight = 0.0;
      while (lo <= hi) {
        mid = lo + (hi - lo) / 2;
        if (x < xdi[mid]) {
          hi = mid - 1;
        } else if (x >= xdi[mid + 1]) {
          lo = mid + 1;
        } else {
          weight = (xdi[mid + 1] - x) / (xdi[mid + 1] - xdi[mid]);
          break;
        }
      }
    }

    indices[i] = mid;
    weights[i] = weight;
  }
}

extern "C"
{

// A generalized version of mlinterp that will work for any
// number of dimensions. For some reason, the code is slower
// than the C++ template, so I use it to extend to high dimensions
int interp_general(
  int ndim, const int *nd, double **xd, const double *yd,
  int ni, const double *xi, double *yi
)
{
  int dimension = ndim;
  int power = 1 << dimension;

  int *indices = (int *) malloc(dimension*sizeof(int));
  if (!indices) {
    return 1;
  }
  double *weights = (double *) malloc(dimension*sizeof(double));
  if (weights == NULL) {
    free(indices);
    return 1;
  }
  int *buffer = (int *) malloc(dimension*sizeof(int));
  if (buffer == NULL) {
    free(indices); 
    free(weights); 
    return 1;
  }
  double factor;

  for (int n = 0; n < ni; ++n){
    yi[n] = 0.0;
    run(
      ndim, nd, xd, 
      n, ni, xi,
      indices, weights
    );
    for (int bitstr = 0; bitstr < power; ++bitstr) {
      factor = 1.0;
      for (int i = 0; i < dimension; ++i) {
        if (bitstr & (1 << i)) {
          buffer[i] = indices[i];
          factor *= weights[i];
        } else {
          buffer[i] = indices[i] + 1;
          factor *= 1 - weights[i];
        }
      }
      if (factor > DBL_MIN) {
        int k = mux(dimension, nd, buffer);
        yi[n] += factor * yd[k];
      }
    }
  }

  free(indices);
  free(weights);
  free(buffer);
  return 0;
}

void interp_wrapper(
  int ndim, const int *nd, double **xd, double *fd,
  int ni, const double *xi, double *fi
)
{

  switch (ndim)
  {
    case 1:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0]
      );
      break;
    case 2:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0*ni], 
        xd[1], &xi[1*ni]
      );
      break;
    case 3:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0*ni], 
        xd[1], &xi[1*ni], 
        xd[2], &xi[2*ni]
      );
      break;
    case 4:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0*ni], 
        xd[1], &xi[1*ni], 
        xd[2], &xi[2*ni], 
        xd[3], &xi[3*ni]
      );
      break;
    case 5:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0*ni], 
        xd[1], &xi[1*ni], 
        xd[2], &xi[2*ni], 
        xd[3], &xi[3*ni], 
        xd[4], &xi[4*ni]
      );
      break;
    case 6:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0*ni], 
        xd[1], &xi[1*ni], 
        xd[2], &xi[2*ni], 
        xd[3], &xi[3*ni], 
        xd[4], &xi[4*ni], 
        xd[5], &xi[5*ni]
      );
      break;
    case 7:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0*ni], 
        xd[1], &xi[1*ni], 
        xd[2], &xi[2*ni], 
        xd[3], &xi[3*ni], 
        xd[4], &xi[4*ni], 
        xd[5], &xi[5*ni], 
        xd[6], &xi[6*ni]
      );
      break;
    case 8:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0*ni], 
        xd[1], &xi[1*ni], 
        xd[2], &xi[2*ni], 
        xd[3], &xi[3*ni], 
        xd[4], &xi[4*ni], 
        xd[5], &xi[5*ni], 
        xd[6], &xi[6*ni], 
        xd[7], &xi[7*ni]
      );
      break;
    case 9:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0*ni], 
        xd[1], &xi[1*ni], 
        xd[2], &xi[2*ni], 
        xd[3], &xi[3*ni], 
        xd[4], &xi[4*ni], 
        xd[5], &xi[5*ni], 
        xd[6], &xi[6*ni], 
        xd[7], &xi[7*ni], 
        xd[8], &xi[8*ni]
      );
      break;
    case 10:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0*ni], 
        xd[1], &xi[1*ni], 
        xd[2], &xi[2*ni], 
        xd[3], &xi[3*ni], 
        xd[4], &xi[4*ni], 
        xd[5], &xi[5*ni], 
        xd[6], &xi[6*ni], 
        xd[7], &xi[7*ni], 
        xd[8], &xi[8*ni], 
        xd[9], &xi[9*ni]
      );
      break;
    default:
      int ierr = interp_general(
        ndim, nd, xd, fd,
        ni, xi, fi
      );
  }

  // Check for out of bounds
  for (int j = 0; j < ni; j++) {
    for (int i = 0; i < ndim; i++) {
      double xij = xi[j + i*ni];
      if (xij < xd[i][0] or xij > xd[i][nd[i]-1])
      {
        fi[j] = NAN;
        break;
      }
    }
  }

}

}