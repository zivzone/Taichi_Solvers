import numpy as np
import taichi as ti

#ti.cfg.arch = ti.x86_64
ti.cfg.arch = ti.cuda

# grid parameters
n_x = 128
n_y = 128
n_z = 128

w_x = 1.0
w_y = 1.0
w_z = 1.0

n_mg_levels = 3

n_ghost = 1 # needs to be atleast 1

# some other constants
Czero = 1.0e-6
Cone = 1.0-Czero

dx = w_x/n_x
dy = w_y/n_y
dz = w_z/n_z

# setup sparse simulation data arrays
real = ti.f32
scalar = lambda: ti.var(dt=real)

U = ti.Vector(n_mg_levels, dt=real) # x velocity on left face
#V = scalar() # y velocity on bottom face
#W = scalar() # z velocity on back face
#P = scalar() # pressure

@ti.layout
def place():
  for l in range(n_mg_levels):
    ti.root.dense(ti.ijk, n_x).place(U(l))

def initialize_kernel(l):
  @ti.kernel
  def kernel():
    for i,j,k in U(l):
      U(l)[i,j,k] = 1.0

  kernel.materialize()
  return kernel

def main():
  # make kernels for each multigrid level
  initialize = np.zeros(n_mg_levels, dtype=ti.Kernel)
  for l in range(n_mg_levels):
    initialize[l] = initialize_kernel(l)

  # run kernels on each multigrid level
  for l in range(n_mg_levels):
    initialize[l]()

  ti.profiler_print()

if __name__ == '__main__':

  main()
