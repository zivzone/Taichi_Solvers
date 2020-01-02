import numpy as np
import taichi as ti

ti.get_runtime().set_default_fp(ti.f64)

ti.cfg.arch = ti.x86_64
#ti.cfg.arch = ti.cuda

# grid parameters
# ******************************************************************************

# internel grid size
nx = 128
ny = 128
nz = 4

# domain dimensions
wx = 64
wy = 64
wz = 4

b_size = 4
sb_size = b_size*4
n_init_subcells = 32

# initial phi params
init_phi = 1 # 0 = zalesaks disk, 1 = cylinder
init_center = [12.0, 12.0 , 0.0]
init_width = 5.0
init_height = 15.0
init_radius = 10.0
init_plane_dir = [1.0, 0.0, 0.0]

# some other constants
Czero = 1.0e-12
Cone = 1.0-Czero
Czero_cleanup = 5.0*Czero
Cone_cleanup = 1.0-Czero_cleanup
small = 1.0e-12
big  = 1.0e12

# computed paramters
n_ghost = 1 # vof requires 1 ghost cell

nx_ext = nx//2 # number of cells exterior to domain on each side. includes ghost cells
ny_ext = ny//2
nz_ext = nz//2

nx_tot = 2*nx;
ny_tot = 2*ny;
nz_tot = 2*nz;

dx = wx/nx
dy = wy/ny
dz = wz/nz
vol = dx*dy*dz

# setup sparse simulation data arrays
# *****************************************************************************
real = ti.f64
iscalar = lambda: ti.var(dt=ti.i32)
scalar = lambda: ti.var(dt=real)
vector = lambda: ti.Vector(3, dt=real)
matrix = lambda: ti.Matrix(3, 3, dt=real)

Flags = iscalar()   # cell, face, vertex flags
C = scalar()        # cell volume fraction
M = vector()        # interface normal vector
Alpha = scalar()    # interface offset
U = scalar()        # x velocity on left face
V = scalar()        # y velocity on bottom face
W = scalar()        # z velocity on back face
Vel_vert = vector() # velocity vector on left/bottom/back vertex
Vert_loc = vector() # position vector of DMC backtracked vertex
Phi = scalar()      # level set at cell center
DCx = scalar()      # delta volume fraction on left face
DCy = scalar()      # delta volume fraction on bottom face
DCz = scalar()      # delta volume fraction on back face
DCx_b = scalar()    # delta volume fraction on left face, used for bounding procedure
DCy_b = scalar()    # delta volume fraction on bottom face, used for bounding procedure
DCz_b = scalar()    # delta volume fraction on back face, used for bounding procedure
Dt = scalar()       # delta t

Flags_temp = iscalar()
C_temp = scalar()

@ti.layout
def data():
  block = ti.root.dense(ti.ijk, [nx_tot//b_size, ny_tot//b_size, nz_tot//b_size]).bitmasked()
  for f in [Flags, C, M, Alpha, U, V, W, Vel_vert, Vert_loc, Phi, DCx, DCy, DCz, DCx_b, DCy_b, DCz_b]:
    block.dense(ti.ijk, b_size).place(f)

  block = ti.root.dense(ti.ijk, [nx_tot//b_size, ny_tot//b_size, nz_tot//b_size]).bitmasked()
  for f in [Flags_temp, C_temp]:
    block.dense(ti.ijk, b_size).place(f)

  ti.root.place(Dt)
