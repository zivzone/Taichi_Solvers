from vof_data import *
from vof_util import *

@ti.kernel
def apply_Neumann_BC():
  for i,j,k in Flags:
    if is_internal_cell(i,j,k) == False:
      di = 0
      if i < nx_ext:
        di  = nx_ext-i
      elif i >= 3*nx_ext:
        di = 3*nx_ext-i-1

      dj = 0
      if j < ny_ext:
        dj  = ny_ext-j
      elif j >= 3*ny_ext:
        dj = 3*ny_ext-j-1

      dk = 0
      if k < nz_ext:
        dk  = nz_ext-k
      elif k >= 3*nz_ext:
        dk = 3*nz_ext-k-1

      C[i,j,k] = C[i+di,j+dj,k+dk]
