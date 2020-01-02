from enum import IntFlag, auto
import numpy as np
from vof_data import *

## location functions
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
  u =  -1.0
  v =  -1.0
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

## level set reconstruction functions ##
@ti.func
def get_phi_from_plic(x,y,z,i,j,k):
  # phi is distance fromm plic plane
  # loc relative to interface cell origin
  x0,y0,z0 = get_vert_loc(i,j,k)
  xt = x-x0
  yt = y-y0
  zt = z-z0

  # phi is distance from plic plane
  phi = -(M[i,j,k][0]*xt + M[i,j,k][1]*yt + M[i,j,k][2]*zt - Alpha[i,j,k]) #\
  #/ti.sqrt(M[i,j,k][0]*M[i,j,k][0] + M[i,j,k][1]*M[i,j,k][1] + M[i,j,k][2]*M[i,j,k][2])
  return phi

@ti.func
def get_phi_from_plic_smooth(x,y,z,i,j,k):
  # get the "smoothed" distance from plic plane
  # ref: 'Single-step reinitialization and extending algorithms for
  # level-set based multi-phase flow simulations'

  # loc relative to interface cell origin
  x0,y0,z0 = get_vert_loc(i,j,k)
  xt = x-x0
  yt = y-y0
  zt = z-z0

  # interface cell center
  xc = dx/2.0
  yc = dy/2.0
  zc = dz/2.0

  #compute the interface point
  phi_c =  -(M[i,j,k][0]*xc + M[i,j,k][1]*yc + M[i,j,k][2]*zc - Alpha[i,j,k]) # M is normalized to length of 1
  xint = xc+phi_c*M[i,j,k][0]
  yint = yc+phi_c*M[i,j,k][1]
  zint = zc+phi_c*M[i,j,k][2]

  # signed distance of interface patch to target point
  d = -(M[i,j,k][0]*xt + M[i,j,k][1]*yt + M[i,j,k][2]*zt - Alpha[i,j,k]) # M is normalized to length of 1

  # signed distance from interface point to target point
  r = ti.sqrt((xint-xt)*(xint-xt)+(yint-yt)*(yint-yt)+(zint-zt)*(zint-zt))
  if d < 0.0: r = -r

  #misalignment length between ray cast from interface point to target point
  h = abs(r*r-d*d)

  # blend d and h with wendland function
  delta = np.sqrt(dx*dx+dy*dy+dz*dz)
  zeta = min(h/(3.0*delta),1.0)
  beta = (1.0-zeta)**6 * (1.0 + 6.0*zeta + 35.0/3.0*zeta**2)

  phi = beta*d + (1.0-beta)*r
  return phi


## data clearing functions ##
@ti.kernel
def clear_data():
  for i,j,k in Flags:
    Flags[i,j,k] = 0
    C[i,j,k] = 0.0
    M[i,j,k] = [0.0, 0.0, 0.0]
    Phi[i,j,k] = 0.0


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
def calc_vol_frac_simple(phi,grad_phi):
  # estimate volume fraction of cell using level set and its gradient
  phim = -ti.abs(phi)

  abs_grad_phi = [0.0,0.0,0.0]
  abs_grad_phi[0] = ti.abs(grad_phi[0])
  abs_grad_phi[1] = ti.abs(grad_phi[1])
  abs_grad_phi[2] = ti.abs(grad_phi[2])
  dxi   = ti.max(abs_grad_phi[0],ti.max(abs_grad_phi[1],abs_grad_phi[2]))
  dzeta = ti.min(abs_grad_phi[0],ti.min(abs_grad_phi[1],abs_grad_phi[2]))
  deta  = abs_grad_phi[0] + abs_grad_phi[1] + abs_grad_phi[2] - dxi - dzeta

  a = ti.max(phim + 0.5 * ( dxi + deta + dzeta), 0.0);
  b = ti.max(phim + 0.5 * ( dxi + deta - dzeta), 0.0);
  c = ti.max(phim + 0.5 * ( dxi - deta + dzeta), 0.0);
  d = ti.max(phim + 0.5 * (-dxi + deta + dzeta), 0.0);
  e = ti.max(phim + 0.5 * ( dxi - deta - dzeta), 0.0);
  vf = 0.0
  if dxi > Czero :
    if deta > Czero :
      if dzeta > Czero : #  3D
        vf = (a*a*a - b*b*b - c*c*c - d*d*d + e*e*e)/(6.0*dxi*deta*dzeta)
      else:#  2D
        vf = (a*a - c*c)/(2.0*dxi*deta)
    else: #  1D
      vf = a/dxi
  else:  #  0D
    if phim !=  0.0:
      vf = 0.0
    else:
      vf = 0.5

  if phi > 0.0 :
    vf = 1.0 - vf

  return vf

## cell identifier functions
@ti.func
def is_internal_cell(i,j,k):
  return (i>=nx_ext and i<nx_tot-nx_ext \
    and j>=ny_ext and j<ny_tot-ny_ext \
    and k>=nz_ext and k<nz_tot-nz_ext)

def is_ghost_cell(i,j,k):
  return is_internal_cell(i,j,k)!= True #(i>nx_ext-1-n_ghost and i<nx_tot-nx_ext+n_ghost \
    #and j>ny_ext-1-n_ghost and j<ny_tot-ny_ext+n_ghost \
    #and k>nz_ext-1-n_ghost and k<nz_tot-nz_ext+n_ghost) \
    #and not is_internal_cell(i,j,k)

@ti.func
def is_internal_vertex(i,j,k):
  return (i>=nx_ext and i<nx_tot-nx_ext+1 \
    and j>=ny_ext and j<ny_tot-ny_ext+1 \
    and k>=nz_ext  and k<nz_tot-nz_ext+1)

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
