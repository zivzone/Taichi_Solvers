import taichi as ti

ti.get_runtime().set_default_fp(ti.f64)

ti.cfg.arch = ti.x86_64
#ti.cfg.arch = ti.cuda

# grid parameters
# ******************************************************************************
CFL = .24
t_final = 3.0
plot_interval = 20

# internel grid size
nx = 128
ny = 128
nz = 4

# domain dimensions
wx = 5.0
wy = 5.0
wz = .0001

# grig blocking params
b_size = 4
sb_size = b_size*4

# initial phi params
n_init_subcells = 2
init_phi = 0 # 0 = zalesaks disk, 1 = cylinder, 2=sphere, 3=plane
init_center = [4.0, 4.0 , 0]
init_width = .42
init_height = 1.0
init_radius = .75
init_plane_dir = [.1, 1.0, 0.0]

# trasport velocity
init_vel = 2 # 0 = rotation, 1 = vortex in a box, 2 = transport
u_transport = -1.0
v_transport = -1.0

# some other constants
small = 1.0e-15
big  = 1.0e15
c_zero = 1.0e-6
c_one = 1.0-c_zero

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
Dt = scalar()       # delta t
Tot_vol = scalar()

Flags_temp = iscalar()
C_temp = scalar()

@ti.layout
def data():
  # sparse blocked layout
  #block = ti.root.dense(ti.ijk, [nx_tot//b_size, ny_tot//b_size, nz_tot//b_size]).bitmasked()
  #for f in [Flags, C, M, Alpha, U, V, W, Vel_vert, Vert_loc, Phi, DCx, DCy, DCz, DCx_b, DCy_b, DCz_b]:
  #  block.dense(ti.ijk, b_size).place(f)

  #block = ti.root.dense(ti.ijk, [nx_tot//b_size, ny_tot//b_size, nz_tot//b_size]).bitmasked()
  #for f in [Flags_temp, C_temp]:
  #  block.dense(ti.ijk, b_size).place(f)

  # dense array layout
  for f in [Flags, C, M, Alpha, U, V, W, Vel_vert, Vert_loc, Phi, DCx, DCy, DCz]:
    ti.root.dense(ti.ijk, [nx_tot, ny_tot, nz_tot]).place(f)

  for f in [Flags_temp, C_temp]:
    ti.root.dense(ti.ijk, [nx_tot, ny_tot, nz_tot]).place(f)

  ti.root.place(Dt,Tot_vol)
