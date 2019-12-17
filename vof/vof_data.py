import numpy as np
import taichi as ti

ti.cfg.arch = ti.x86_64
#ti.cfg.arch = ti.cuda

# grid parameters
# ******************************************************************************

# internel grid size
nx = 512
ny = 512
nz = 2

# domain dimensions
wx = 1000.0
wy = 1000.0
wz = 4.0

b_size = 4
sb_size = b_size*4
n_init_subcells = 4

# initial phi params
init_phi = 0 # 0 = zalesaks disk, 1 = cylinder
init_center = [500.0, 800.0 , 0.0]
init_width = 25
init_height = 150
init_radius = 100.0
init_plane_dir = [1.0, 0.0, 0.0]

# some other constants
Czero = 1.0e-6
Cone = 1.0-Czero
small = 1.0e-6
big  = 1.0e6

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
# *****************************************************************************
real = ti.f32
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
Vert_pos = vector() # position vector of DMC backtracked vertex
Phi = scalar()      # level set at cell center
DCx = scalar()      # delta volume fraction on left face
DCy = scalar()      # delta volume fraction on bottom face
DCz = scalar()      # delta volume fraction on back face

Flags_temp = iscalar()
C_temp = scalar()

@ti.layout
def data():
  block = ti.root.dense(ti.ijk, [nx_tot//b_size, ny_tot//b_size, nz_tot//b_size]).bitmasked()
  for f in [Flags, C, M, Alpha, U, V, W, Vel_vert, Vert_pos, Phi, DCx, DCy, DCz]:
    block.dense(ti.ijk, b_size).place(f)

  block = ti.root.dense(ti.ijk, [nx_tot//b_size, ny_tot//b_size, nz_tot//b_size]).bitmasked()
  for f in [Flags_temp, C_temp]:
    block.dense(ti.ijk, b_size).place(f)
