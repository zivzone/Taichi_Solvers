from enum import IntFlag, auto
import numpy as np
import taichi as ti

#ti.cfg.arch = ti.x86_64
ti.cfg.arch = ti.cuda

# grid parameters
n_x = 512
n_y = 512
n_z = 4

w_x = 1.0
w_y = 1.0
w_z = .01

n_ghost = 0
block_size = 4
n_init_subcells = 1 # must be power of 2

init_phi = 0 # 0 = zalesaks disk, 1 = cylinder
init_center = [.5, .5 , .5]
init_width = .05
init_height = .12
init_radius = .125

dx = w_x/n_x
dy = w_y/n_y
dz = w_z/n_z

class cellFlags(IntFlag):
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

Flags = iscalar()
C = scalar()
M = vector()
Alpha = scalar()
U = vector()
U_vert = vector()
dC = vector()

Flags_temp = iscalar()
C_temp = scalar()
M_temp = vector()
Alpha_temp = scalar()
U_temp = vector()
U_vert_temp = vector()
dC_temp = vector()

@ti.layout
def place():
	block = ti.root.dense(ti.ijk, [n_x//block_size, n_y//block_size, n_z//block_size]).pointer()
	for f in [Flags, C, M, Alpha, U, U_vert, dC]:
		block.dense(ti.ijk, block_size).place(f)

	block_temp = ti.root.dense(ti.ijk, [n_x//block_size, n_y//block_size, n_z//block_size]).pointer()
	for f in [Flags_temp, C_temp, M_temp, Alpha_temp, U_temp, U_vert_temp, dC_temp]:
		block_temp.dense(ti.ijk, block_size).place(f)
