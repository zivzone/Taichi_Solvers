from enum import IntFlag, auto
from vof_data import *


@ti.func
def get_block_loc(ib,jb,kb):
  x = ib*dx*b_size + dx*b_size/2.0 - nx_ext*dx
  y = jb*dy*b_size + dy*b_size/2.0 - ny_ext*dy
  z = kb*dz*b_size + dz*b_size/2.0 - nz_ext*dz
  return x,y,z

@ti.func
def get_cell_loc(i,j,k):
  x = i*dx + dx/2.0 - nx_ext*dx
  y = j*dy + dy/2.0 - ny_ext*dy
  z = k*dz + dz/2.0 - nz_ext*dz
  return x,y,z

@ti.func
def get_vert_loc(i,j,k):
  x = i*dx - nx_ext*dx
  y = j*dy - ny_ext*dy
  z = k*dz - nz_ext*dz
  return x,y,z


## get_velocity functions ##
@ti.func
def get_vel_solid_body_rotation(x,y,z):
  u = 0.5 - y/wy
  v = -0.5 + x/wx
  w = 0.0
  return u,v,w

@ti.func
def get_vel_vortex_in_a_box(x,y,z):
  xpi = np.pi*x/wx
  ypi = np.pi*y/wy
  u = -2.0*ti.sin(xpi)*ti.sin(xpi)*ti.sin(ypi)*ti.cos(ypi)
  v =  2.0*ti.sin(ypi)*ti.sin(ypi)*ti.sin(xpi)*ti.cos(xpi)
  w =  0.0
  return u,v,w

@ti.func
def get_vel_transport(x,y,z):
  u =  1.0
  v =  0.0
  w =  0.0
  return u,v,w

get_vel = get_vel_transport

@ti.kernel
def set_face_velocity():
  # set left/bottom/back face velocities from preset field
  for i,j,k in Flags:
    x,y,z = get_cell_loc(i,j,k)
    u,v,w = get_vel(x-dx/2.0,y,z) # at face loc
    U[i,j,k] = u
    u,v,w = get_vel(x,y-dy/2.0,z) # at face loc
    V[i,j,k] = v
    u,v,w = get_vel(x,y,z-dz/2.0) # at face loc
    W[i,j,k] = w

@ti.func
def get_phi_from_plic(x,y,z,i,j,k):
  # loc relative to interface cell origin
  x0,y0,z0 = get_vert_loc(i,j,k)
  x = x-x0
  y = y-y0
  z = z-z0

  # phi is distance from plic plane
  phi = (M[i,j,k][0]*x + M[i,j,k][1]*y + M[i,j,k][2]*z - Alpha[i,j,k])\
  /ti.sqrt(M[i,j,k][0]*M[i,j,k][0] + M[i,j,k][1]*M[i,j,k][1] + M[i,j,k][2]*M[i,j,k][2])

  return phi

@ti.func
def get_phi_from_plic_smooth(x,y,z,i,j,k):
  # loc relative to interface cell origin
  x0,y0,z0 = get_vert_loc(i,j,k)
  x = x-x0
  y = y-y0
  z = z-z0

  # phi is distance from plic plane
  phi = (M[i,j,k][0]*x + M[i,j,k][1]*y + M[i,j,k][2]*z - Alpha[i,j,k]) \
  /ti.sqrt(M[i,j,k][0]*M[i,j,k][0] + M[i,j,k][1]*M[i,j,k][1] + M[i,j,k][2]*M[i,j,k][2])
  return phi


@ti.kernel
def clear_data():
  for i,j,k in Flags:
    Flags[i,j,k] = 0
    C[i,j,k] = 0.0


@ti.kernel
def clear_data_temp():
  for i,j,k in Flags_temp:
    Flags_temp[i,j,k] = 0
    C_temp[i,j,k] = 0.0


def clear_data_and_deactivate():
  Flags.ptr.snode().parent.parent.clear_data_and_deactivate()

def clear_data_and_deactivate_temp():
  Flags_temp.ptr.snode().parent.parent.clear_data_and_deactivate()


@ti.func
def is_internal_cell(i,j,k):
  return (i>nx_ext-1 and i<nx_tot-nx_ext \
    and j>ny_ext-1 and j<ny_tot-ny_ext \
    and k>nz_ext-1 and k<nz_tot-nz_ext)

def is_ghost_cell(i,j,k):
  return (i>nx_ext-1-n_ghost and i<nx_tot-nx_ext+n_ghost \
    and j>ny_ext-1-n_ghost and j<ny_tot-ny_ext+n_ghost \
    and k>nz_ext-1-n_ghost and k<nz_tot-nz_ext+n_ghost) \
    and not is_internal_cell(i,j,k)

@ti.func
def is_internal_vertex(i,j,k):
  return (i>nx_ext-1 and i<nx_tot-nx_ext+1 \
    and j>ny_ext-1 and j<ny_tot-ny_ext+1 \
    and k>nz_ext-1  and k<nz_tot-nz_ext+1)

@ti.func
def is_ghost_vertex(i,j,k):
  return (i<nx_ext-n_ghost and i>nx_tot-nx_ext+1+n_ghost \
    and j>ny_ext-1-n_ghost and j<ny_tot-ny_ext+1+n_ghost \
    and k>nz_ext-1-n_ghost and k<nz_tot-nz_ext+1+n_ghost)

@ti.func
def is_internal_x_face(i,j,k):
  return (i>nx_ext and i<nx_tot-nx_ext-1 \
    and j>ny_ext-1 and j<ny_tot-ny_ext \
    and k>nz_ext-1 and k<nz_tot-nz_ext)

@ti.func
def is_boundary_x_face(i,j,k):
  return ((i==nx_ext or i==nx_tot-nx_ext-1) \
    and j>ny_ext-1 and j<ny_tot-ny_ext \
    and k>nz_ext-1 and k<nz_tot-nz_ext)

@ti.func
def is_internal_y_face(i,j,k):
  return (i>nx_ext-1 and i<nx_tot-nx_ext \
    and j>ny_ext and j<ny_tot-ny_ext-1 \
    and k>nz_ext-1 and k<nz_tot-nz_ext)

@ti.func
def is_boundary_y_face(i,j,k):
  return (i>nx_ext-1 and i<nx_tot-nx_ext \
    and (j==ny_ext or j==ny_tot-ny_ext-1) \
    and k>nz_ext-1 and k<nz_tot-nz_ext)

@ti.func
def is_internal_z_face(i,j,k):
  return (i>nx_ext-1 and i<nx_tot-nx_ext \
    and j>ny_ext-1 and j<ny_tot-ny_ext \
    and k>nz_ext and k<nz_tot-nz_ext-1)

@ti.func
def is_boundary_z_face(i,j,k):
  return (i>nx_ext-1 and i<nx_tot-nx_ext \
    and j>ny_ext-1 and j<ny_tot-ny_ext \
    and (k==nz_ext and k==nz_tot-nz_ext-1))

class flag_enum(IntFlag):
  NONE = 0
  CELL_ACTIVE = auto()
  CELL_INTERFACE = auto()
  CELL_BUFFER = auto()
  X_FACE_ACTIVE = auto()
  Y_FACE_ACTIVE = auto()
  Z_FACE_ACTIVE = auto()

@ti.func
def is_interface_cell(i,j,k):
  return Flags[i,j,k]&flag_enum.CELL_INTERFACE==flag_enum.CELL_INTERFACE

@ti.func
def is_active_cell(i,j,k):
  return Flags[i,j,k]&flag_enum.CELL_ACTIVE==flag_enum.CELL_ACTIVE

@ti.func
def is_buffer_cell(i,j,k):
  return Flags[i,j,k]&flag_enum.CELL_BUFFER==flag_enum.CELL_BUFFER

@ti.func
def is_active_x_face(i,j,k):
  return Flags[i,j,k]&flag_enum.X_FACE_ACTIVE==flag_enum.X_FACE_ACTIVE

@ti.func
def is_active_y_face(i,j,k):
  return Flags[i,j,k]&flag_enum.Y_FACE_ACTIVE==flag_enum.Y_FACE_ACTIVE

@ti.func
def is_active_z_face(i,j,k):
  return Flags[i,j,k]&flag_enum.Z_FACE_ACTIVE==flag_enum.Z_FACE_ACTIVE