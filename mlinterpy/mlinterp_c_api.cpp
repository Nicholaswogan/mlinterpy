#include <math.h>
#include "mlinterp.hpp"
using namespace mlinterp;

extern "C"
{

void interp_single(
  int n, int *nd, double **xd, double *fd,
  double *xi, double *fi
)
{

  // First, check for out of bounds
  for (int i = 0; i < n; i++)
  {
    if (xi[i] < xd[i][0] or xi[i] > xd[i][nd[i]-1])
    {
      *fi = NAN;
      return;
    }
  }

  // Next, interpolate
  const int ni = 1;
  switch (n)
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
        xd[0], &xi[0], 
        xd[1], &xi[1]
      );
      break;
    case 3:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0], 
        xd[1], &xi[1], 
        xd[2], &xi[2]
      );
      break;
    case 4:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0], 
        xd[1], &xi[1], 
        xd[2], &xi[2], 
        xd[3], &xi[3]
      );
      break;
    case 5:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0], 
        xd[1], &xi[1], 
        xd[2], &xi[2], 
        xd[3], &xi[3], 
        xd[4], &xi[4]
      );
      break;
    case 6:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0], 
        xd[1], &xi[1], 
        xd[2], &xi[2], 
        xd[3], &xi[3], 
        xd[4], &xi[4], 
        xd[5], &xi[5]
      );
      break;
    case 7:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0], 
        xd[1], &xi[1], 
        xd[2], &xi[2], 
        xd[3], &xi[3], 
        xd[4], &xi[4], 
        xd[5], &xi[5], 
        xd[6], &xi[6]
      );
      break;
    case 8:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0], 
        xd[1], &xi[1], 
        xd[2], &xi[2], 
        xd[3], &xi[3], 
        xd[4], &xi[4], 
        xd[5], &xi[5], 
        xd[6], &xi[6], 
        xd[7], &xi[7]
      );
      break;
    case 9:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0], 
        xd[1], &xi[1], 
        xd[2], &xi[2], 
        xd[3], &xi[3], 
        xd[4], &xi[4], 
        xd[5], &xi[5], 
        xd[6], &xi[6], 
        xd[7], &xi[7], 
        xd[8], &xi[8]
      );
      break;
    case 10:
      interp(
        nd, ni,
        fd, fi,
        xd[0], &xi[0], 
        xd[1], &xi[1], 
        xd[2], &xi[2], 
        xd[3], &xi[3], 
        xd[4], &xi[4], 
        xd[5], &xi[5], 
        xd[6], &xi[6], 
        xd[7], &xi[7], 
        xd[8], &xi[8], 
        xd[9], &xi[9]
      );
      break;
  }
}

void interp_vector(
  int n, int *nd, double **xd, double *fd,
  int ni, double *xi, double *fi
)
{

  for (int i = 0; i < ni; i++)
  {
    interp_single(n, nd, xd, fd, &xi[i*n], &fi[i]);
  }

}

}