import numpy as np
import taichi as ti

ti.cfg.arch = ti.x86_64
ti.cfg.arch = ti.cuda

# grid parameters
n_mg_levels = 3
pre_and_post_smoothing = 2
bottom_smoothing = 50

nx = 64
ny = 64
nz = 64

wx = 1.0
wy = 1.0
wz = 1.0

nx_ghost = nx//2 # number of ghost cells set so that that total grid size is still power of 2
ny_ghost = ny//2
nz_ghost = nz//2

nx_tot = 2*nx;
ny_tot = 2*ny;
nz_tot = 2*nz;

dx = wx/nx
dy = wy/ny
dz = wz/nz

# setup sparse simulation data arrays
real = ti.f32

R     = ti.Vector(n_mg_levels, dt=real)
Z     = ti.Vector(n_mg_levels, dt=real)
X     = ti.var(dt=real)
P     = ti.var(dt=real)
AP    = ti.var(dt=real)
Alpha = ti.var(dt=real)
Beta  = ti.var(dt=real)
Sum   = ti.var(dt=real)
Phase = ti.var(dt=real)

@ti.layout
def place():
  grid = ti.root.dense(ti.ijk, [nx_tot//4, ny_tot//4, nz_tot//4]).bitmasked().dense(ti.ijk,4)

  for l in range(n_mg_levels):
    grid.place(R(l))
    grid.place(Z(l))

  grid.place(X)
  grid.place(P)
  grid.place(AP)

  ti.root.place(Alpha, Beta, Sum, Phase)


@ti.kernel
def compute_AP():
  for i,j,k in AP:
    AP[i,j,k] = 6.0 * P[i,j,k] - P[i-1,j,k] - P[i+1,j,k] \
              - P[i,j-1,k] - P[i,j-1,k] - P[i,j,k+1] \
              - P[i,j,k-1]


@ti.kernel
def reduce_R():
  for i,j,k in P:
    Sum[None] += R(0)[i,j,k]*R(0)[i,j,k]

@ti.kernel
def reduce_ZTR():
  for i,j,k in P:
    Sum[None] += Z(0)[i,j,k]*R(0)[i,j,k]

@ti.kernel
def reduce_PAP():
  for i,j,k in P:
    Sum[None] += P[i,j,k]*AP[i,j,k]

@ti.kernel
def update_X():
  for i,j,k in X:
    X[i,j,k] += Alpha[None]*P[i,j,k]

@ti.kernel
def update_R():
  for i,j,k in P:
    R(0)[i,j,k] -= Alpha[None]*AP[i,j,k]

@ti.kernel
def update_P():
  for i,j,k in P:
    P[i,j,k] = Z(0)[i,j,k] + Beta[None]*P[i,j,k]

@ti.kernel
def init_R():
  for i,j,k in P:
    x = (i-nx_tot/4)*2.0/nx_tot
    y = (j-ny_tot/4)*2.0/ny_tot
    z = (k-nz_tot/4)*2.0/nz_tot
    R(0)[i,j,k] = sin(2.0*np.pi*x) * sin(2.0*np.pi*y) * sin(2.0*np.pi*z)

def make_restrict(l):
  @ti.kernel
  def kernel():
    for i,j,k in R(l):
      res = R(l)[i,j,k] - (6.0 * Z(l)[i,j,k]) \
                        - Z(l)[i-1,j,k] - Z(l)[i+1,j,k] \
                        - Z(l)[i,j-1,k] - Z(l)[i,j+1,k] \
                        - Z(l)[i,j,k-1] - Z(l)[i,j,k+1]
      R(l+1)[i/2,j/2,k/2] += res*0.5
  kernel.materialize()
  return kernel

def make_prolongate(l):
  @ti.kernel
  def kernel():
    for i,j,k in Z(l):
      Z(l)[i,j,k] += Z(l+1)[i/2,j/2,k/2]
  kernel.materialize()
  return kernel

def make_smooth(l):
  @ti.kernel
  def kernel():
    for i,j,k in Z(l):
      ret = 0.0
      if (i+j+k)&1 == Phase[None]:
        ret = 1.0/6.0*(R(l)[i,j,k] + Z(l)[i-1,j,k] + Z(l)[i+1,j,k] \
                          + Z(l)[i,j-1,k] + Z(l)[i,j+1,k] \
                          + Z(l)[i,j,k-1] + Z(l)[i,j,k+1]) \
                          - Z(l)[i,j,k]
      Z(l)[i,j,k] += ret
  kernel.materialize()
  return kernel

def make_clear_R(l):
  @ti.kernel
  def kernel():
    for i,j,k in R(l):
      R(l)[i,j,k] = 0.0
  kernel.materialize()
  return kernel

def make_clear_Z(l):
  @ti.kernel
  def kernel():
    for i,j,k in Z(l):
      Z(l)[i,j,k] = 0.0
  kernel.materialize()
  return kernel

# make kernels for each multigrid level
restrict = np.zeros(n_mg_levels-1, dtype=ti.Kernel)
prolongate = np.zeros(n_mg_levels-1, dtype=ti.Kernel)
for l in range(n_mg_levels-1):
  restrict[l] = make_restrict(l)
  prolongate[l] = make_prolongate(l)

smooth = np.zeros(n_mg_levels, dtype=ti.Kernel)
clear_R = np.zeros(n_mg_levels, dtype=ti.Kernel)
clear_Z = np.zeros(n_mg_levels, dtype=ti.Kernel)
for l in range(n_mg_levels):
  smooth[l] = make_smooth(l)
  clear_R[l] = make_clear_R(l)
  clear_Z[l] = make_clear_Z(l)

@ti.kernel
def identity():
  for i,j,k in Z(0):
    Z(0)[i,j,k] = R(0)[i,j,k]


def apply_preconditioner():
  clear_Z[0]()
  for l in range(n_mg_levels-1):
    for i in range(pre_and_post_smoothing << l):
      Phase[None] = 0
      smooth[l]()
      Phase[None] = 1
      smooth[l]()
    clear_Z[l+1]()
    clear_R[l+1]()
    restrict[l]()

  for i in range(bottom_smoothing):
    Phase[None] = 0
    smooth[n_mg_levels-1]()
    Phase[None] = 1
    smooth[n_mg_levels-1]()

  for l in reversed(range(n_mg_levels-1)):
    prolongate[l]()
    for i in range(pre_and_post_smoothing << l):
      Phase[None] = 1
      smooth[l]()
      Phase[None] = 0
      smooth[l]()



Sum[None] = 0.0
reduce_R()
initial_RTR = Sum[None]


apply_preconditioner()
update_P()
Sum[None] = 0.0
reduce_ZTR()
old_ZTR = Sum[None]

# CG
for i in range(400):
  compute_AP()
  Sum[None] = 0.0
  reduce_PAP()
  PAP = Sum[None]
  # alpha = rTr / pTAp
  Alpha[None] = old_ZTR/PAP

  # x = x + alpha p
  update_X()
  # r = r - alpha Ap
  update_R()

  # z = M^-1 r
  apply_preconditioner()
  Sum[None] = 0.0
  reduce_ZTR()
  new_ZTR = Sum[None]

  Sum[None] = 0.0
  reduce_R()
  RTR = Sum[None]
  if RTR < initial_RTR*1.0e-12:
    break
  # beta = new_RTR / old_RTR
  Beta[None] = new_ZTR / old_ZTR

  # p = z + beta p
  update_P()
  old_ZTR = new_ZTR

  print(i)
