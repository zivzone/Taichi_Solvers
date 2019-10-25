from enum import IntFlag, auto
import numpy as np
import taichi as ti

#ti.cfg.arch = ti.x86_64
ti.cfg.arch = ti.cuda

# grid parameters

# internel grid size
nx = 256
ny = 256
nz = 16

# domain dimensions
wx = 1000.0
wy = 1000.0
wz = 80

b_size = 4
sb_size = b_size*4
n_init_subcells = 8

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

nx_ghost = nx//2 # number of ghost cells set so that that total grid size is still power of 2
ny_ghost = ny//2
nz_ghost = nz//2

nx_tot = 2*nx;
ny_tot = 2*ny;
nz_tot = 2*nz;

dx = wx/nx
dy = wy/ny
dz = wz/nz

class flag_enum(IntFlag):
  NONE = 0
  CELL_ACTIVE = auto()
  CELL_INTERFACE = auto()
  CELL_GHOST = auto()
  X_FACE_ACTIVE = auto()
  Y_FACE_ACTIVE = auto()
  Z_FACE_ACTIVE = auto()

# setup sparse simulation data arrays
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
Phi = scalar()      # level set at cell center
DCx = scalar()      # delta volume fraction on left face
DCy = scalar()      # delta volume fraction on bottom face
DCz = scalar()      # delta volume fraction on back face

Flags_temp = iscalar()
C_temp = scalar()

@ti.layout
def place():
  super_block = ti.root.dense(ti.ijk, [nx_tot//sb_size, ny_tot//sb_size, nz_tot//sb_size]).pointer()
  block = super_block.dense(ti.ijk, [sb_size//b_size, sb_size//b_size, sb_size//b_size]).pointer()
  for f in [Flags, C, M, Alpha, U, V, W, Vel_vert, Phi, DCx, DCy, DCz]:
    block.dense(ti.ijk, b_size).place(f)

  super_block = ti.root.dense(ti.ijk, [nx_tot//sb_size, ny_tot//sb_size, nz_tot//sb_size]).pointer()
  block = super_block.dense(ti.ijk, [sb_size//b_size, sb_size//b_size, sb_size//b_size]).pointer()
  for f in [Flags_temp, C_temp]:
    block.dense(ti.ijk, b_size).place(f)
