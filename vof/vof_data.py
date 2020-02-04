import taichi as ti

ti.get_runtime().set_default_fp(ti.f64)

#ti.cfg.arch = ti.x86_64
ti.cfg.arch = ti.cuda

# grid parameters
# ******************************************************************************
CFL = .25
t_final = 10
plot_interval = 20

# internel grid size
nx = 256
ny = 256
nz = 4

# domain dimensions
wx = 5.0
wy = 5.0
wz = .01

# grig blocking params
b_size = 4
sb_size = b_size*4

# initial phi params
n_init_subcells = 2
init_phi = 1 # 0 = zalesaks disk, 1 = cylinder, 2=sphere, 3=plane
init_center = [2.5, 3.5 , 0]
init_width = .42
init_height = 1.5
init_radius = 1.0
init_plane_dir = [.1, 1.0, 0.0]

# trasport velocity
init_vel = 1 # 0 = rotation, 1 = vortex in a box, 2 = transport
u_transport = 1.0
v_transport = 1.0

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

"""
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
  block = ti.root.dense(ti.ijk, [nx_tot//b_size, ny_tot//b_size, nz_tot//b_size]).pointer()
  for f in [Flags, C, M, Alpha, U, V, W, Phi, DCx, DCy, DCz]:
    block.dense(ti.ijk, b_size).place(f)

  block = ti.root.dense(ti.ijk, [nx_tot//b_size, ny_tot//b_size, nz_tot//b_size]).pointer()
  for f in [Flags_temp, C_temp]:
    block.dense(ti.ijk, b_size).place(f)

  ti.root.place(Dt,Tot_vol)

@ti.kernel
def clear():
  for i,j,k in Flags.parent().parent():
    ti.deactivate(Flags.parent().parent(), [i,j,k])

@ti.kernel
def clear_temp():
  for i,j,k in Flags_temp.parent().parent():
    ti.deactivate(Flags_temp.parent().parent(), [i,j,k])

"""
## setup dense simulation data arrays
real = ti.f64
vector = lambda: ti.Vector(3, dt=real, shape=(nx_tot,ny_tot,nz_tot))
scalar = lambda: ti.var(dt=real, shape=(nx_tot,ny_tot,nz_tot))
iscalar = lambda: ti.var(dt=ti.i32, shape=(nx_tot,ny_tot,nz_tot))

Flags = iscalar()   # cell, face, vertex flags
C = scalar()        # cell volume fraction
M = vector()        # interface normal vector
Alpha = scalar()    # interface offset
U = scalar()        # x velocity on left face
V = scalar()        # y velocity on bottom face
W = scalar()        # z velocity on back face
Phi = scalar()      # level set at cell center
DCx = scalar()      # delta volume fraction on left face
DCy = scalar()      # delta volume fraction on bottom face
DCz = scalar()      # delta volume fraction on back face
Dt = ti.var(dt=real, shape=())       # delta t
Tot_vol = ti.var(dt=real, shape=())

Flags_temp = iscalar()
C_temp = scalar()

## data clearing functions ##
@ti.kernel
def clear():
  for i,j,k in Flags:
    Flags[i,j,k] = 0
    #C[i,j,k] = 0.0
    M[i,j,k] = [0.0, 0.0, 0.0]
    Phi[i,j,k] = 0.0

@ti.kernel
def clear_temp():
  for i,j,k in Flags_temp:
    Flags_temp[i,j,k] = 0
    #C_temp[i,j,k] = 0.0
