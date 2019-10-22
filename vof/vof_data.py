from enum import IntFlag, auto
import numpy as np
import taichi as ti

#ti.cfg.arch = ti.x86_64
ti.cfg.arch = ti.cuda

# grid parameters
n_x = 512
n_y = 512
n_z = 4

w_x = 1000
w_y = 1000
w_z = 1

n_ghost = 1 # needs to be atleast 1
b_size = 4
sb_size = b_size*1
n_init_subcells = 4

# initial phi params
init_phi = 2 # 0 = zalesaks disk, 1 = cylinder
init_center = [0, 0 , 0]
init_width = .05
init_height = .125
init_radius = 50
init_plane_dir = [1.0, 0.0, 0.0]

# some other constants
Czero = 1.0e-6
Cone = 1.0-Czero

dx = w_x/n_x
dy = w_y/n_y
dz = w_z/n_z

class cell_flags(IntFlag):
	NONE = 0
	CELL_ACTIVE = auto()
	CELL_INTERFACE = auto()
	CELL_GHOST = auto()

# setup sparse simulation data arrays
real = ti.f32
iscalar = lambda: ti.var(dt=ti.i32)
scalar = lambda: ti.var(dt=real)
vector = lambda: ti.Vector(3, dt=real)
matrix = lambda: ti.Matrix(3, 3, dt=real)

Flags = iscalar() # cell, face, vertex flags
C = scalar() # cell volume fraction
M = vector() # interface normal vector
Alpha = scalar() # interface offset
U = scalar() # x velocity on left face
V = scalar() # y velocity on bottom face
W = scalar() # z velocity on back face
Vel_vert = vector() # velocity vector on left/bottom/back vertex
dCx = scalar() # delta volume fraction on left face
dCy = scalar() # delta volume fraction on bottom face
dCz = scalar() # delta volume fraction on back face

Flags_temp = iscalar()
C_temp = scalar()

@ti.layout
def place():
	super_block = ti.root.dense(ti.ijk, [n_x//sb_size, n_y//sb_size, n_z//sb_size]).pointer()
	block = super_block.dense(ti.ijk, [sb_size//b_size, sb_size//b_size, sb_size//b_size]).pointer()
	for f in [Flags, C, M, Alpha, U, V, W, Vel_vert, dCx, dCy, dCz]:
		block.dense(ti.ijk, b_size).place(f)

	super_block = ti.root.dense(ti.ijk, [n_x//sb_size, n_y//sb_size, n_z//sb_size]).pointer()
	block = super_block.dense(ti.ijk, [sb_size//b_size, sb_size//b_size, sb_size//b_size]).pointer()
	for f in [Flags_temp, C_temp]:
		block.dense(ti.ijk, b_size).place(f)

def clear_data_and_deactivate():
	Flags.ptr.snode().parent.parent.parent.clear_data_and_deactivate()

def clear_data_and_deactivate_temp():
	Flags_temp.ptr.snode().parent.parent.parent.clear_data_and_deactivate()

@ti.func
def is_internal(i,j,k):
	return (i>n_ghost-1 and j>n_ghost-1 and k>n_ghost-1 and i<n_x-n_ghost and j<n_y-n_ghost and k<n_z-n_ghost)
