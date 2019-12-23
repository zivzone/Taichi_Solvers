import numpy as np
import taichi as ti

#ti.cfg.arch = ti.x86_64
ti.cfg.arch = ti.cuda

# grid parameters
nx = 128
ny = 128
nz = 128

wx = 1.0
wy = 1.0
wz = 1.0

n_mg_levels = 3

nx_ghost = nx//2 # number of ghost cells set so that that total grid size is still power of 2
ny_ghost = ny//2
nz_ghost = nz//2

dx = wx/nx
dy = wy/ny
dz = wz/nz

# setup sparse simulation data arrays
real = ti.f64

U = ti.Vector(n_mg_levels, dt=real) # x velocity on left face
V = ti.Vector(n_mg_levels, dt=real) # y velocity on bottom face
W = ti.Vector(n_mg_levels, dt=real) # z velocity on back face
P = ti.Vector(n_mg_levels, dt=real) # pressure

@ti.layout
def place():
  for l in range(n_mg_levels):
    ti.root.dense(ti.ijk, nx).place(U(l),V(l),W(l),P(l))

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
