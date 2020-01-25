from enum import IntFlag, auto
import numpy as np
from vof_data import *


## location functions ##
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
  u =  u_transport
  v =  v_transport
  w =  0.0
  return u,v,w

if(init_vel == 0):
  get_vel = get_vel_solid_body_rotation
elif(init_vel == 1):
  get_vel = get_vel_vortex_in_a_box
else:
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
  # loc is relative to interface cell vertex
  x0,y0,z0 = get_vert_loc(i,j,k)
  phi = -(M[i,j,k][0]*(x-x0) + M[i,j,k][1]*(y-y0) + M[i,j,k][2]*(z-z0) - Alpha[i,j,k]) \
        /ti.sqrt(M[i,j,k][0]*M[i,j,k][0] + M[i,j,k][1]*M[i,j,k][1] + M[i,j,k][2]*M[i,j,k][2])

  return phi

@ti.func
def get_phi_and_weight_from_plic(x,y,z,i,j,k):
  # phi is distance fromm plic plane
  # loc is relative to interface cell vertex
  x0,y0,z0 = get_vert_loc(i,j,k)
  xt = x-x0
  yt = y-y0
  zt = z-z0

  phi = -(M[i,j,k][0]*xt + M[i,j,k][1]*yt + M[i,j,k][2]*zt - Alpha[i,j,k]) \
      /ti.sqrt(M[i,j,k][0]*M[i,j,k][0] + M[i,j,k][1]*M[i,j,k][1] + M[i,j,k][2]*M[i,j,k][2])


  # interface cell center
  xc = dx/2.0
  yc = dy/2.0
  zc = dz/2.0

  # interface center point
  phi_c =  -(M[i,j,k][0]*xc + M[i,j,k][1]*yc + M[i,j,k][2]*zc - Alpha[i,j,k]) \
      /ti.sqrt(M[i,j,k][0]*M[i,j,k][0] + M[i,j,k][1]*M[i,j,k][1] + M[i,j,k][2]*M[i,j,k][2])

  xint = xc+phi_c*M[i,j,k][0]
  yint = yc+phi_c*M[i,j,k][1]
  zint = zc+phi_c*M[i,j,k][2]

  # distance of interface point to target point
  r = ti.sqrt((xint-xt)*(xint-xt)+(yint-yt)*(yint-yt)+(zint-zt)*(zint-zt))

  # weight
  w = phi*phi/(r*r)

  return phi, w

#######  plic functions
@ti.func
def calc_plic_from_phi(phi):
  # calculate the plic normal vector m and distance alpha from levelsets at cell vertices

  # levelset and gradient at cell center
  m  = ti.Vector([0.0,0.0,0.0])
  phi_c = (phi[0][0][0]+phi[1][0][0]+phi[0][1][0]+phi[1][1][0] \
          + phi[0][0][1]+phi[1][0][1]+phi[0][1][1]+phi[1][1][1])/8.0
  m[0] = -(phi[0][0][0]-phi[1][0][0]+phi[0][1][0]-phi[1][1][0] \
          + phi[0][0][1]-phi[1][0][1]+phi[0][1][1]-phi[1][1][1])/4.0
  m[1] = -(phi[0][0][0]+phi[1][0][0]-phi[0][1][0]-phi[1][1][0] \
          + phi[0][0][1]+phi[1][0][1]-phi[0][1][1]-phi[1][1][1])/4.0
  m[2] = -(phi[0][0][0]+phi[1][0][0]+phi[0][1][0]+phi[1][1][0] \
          - phi[0][0][1]-phi[1][0][1]-phi[0][1][1]-phi[1][1][1])/4.0

  len = ti.abs(m[0]) + ti.abs(m[1]) + ti.abs(m[2])
  m = -m/len
  alpha = phi_c/len + 0.5*(m[0]+m[1]+m[2]) - min(0.0,m[0]) - min(0.0,m[1]) - min(0.0,m[2])

  return alpha,m


@ti.func
def calc_C(alpha, m):
  # computes the volume fraction given the normal vector and plane constant alpha
  c = 0.0
  if alpha < 0.0:
    c = 0.0
  elif (alpha > (ti.abs(m[0])+ti.abs(m[1])+ti.abs(m[2]))):
    c = 1.0
  else:
    # convert normal vector into Zaleski's m vector
    a = ti.min(alpha, ti.abs(m[0]) + ti.abs(m[1]) + ti.abs(m[2]) - alpha)
    mx = ti.abs(m[0])
    my = ti.abs(m[1])
    mz = ti.abs(m[2])

    # the coefficients of the normal must be ordered as: m1 < m2 < m3
    m1 = ti.min(mx,ti.min(my,mz))
    m3 = ti.max(mx,ti.max(my,mz))
    m2 = mx+my+mz-m1-m3

    m12 = m1 + m2
    mm  = ti.min(m12,m3)
    pr  = ti.max(6.0*m1*m2*m3,1.0e-50)
    V1  = m1*m1*m1/pr

    if (a <  m1):
      c = a*a*a/pr
    elif (a < m2):
      c = 0.5*a*(a-m1)/(m2*m3)+V1
    elif (a < mm):
      c = (a*a*(3.0*m12-a)+m1*m1*(m1-3.0*a)+m2*m2*(m2-3.0*a))/pr
    elif (m12 <= m3):
      c = (a-0.5*m12)/m3
    else:
      c = (a*a*(3.0*(m1+m2+m3)-2.0*a) + m1*m1*(m1-3.0*a) + \
      m2*m2*(m2-3.0*a) + m3*m3*(m3-3.0*a))/pr

    if (alpha > 0.5*(ti.abs(m[0])+ti.abs(m[1])+ti.abs(m[2]))):
      c = 1.0-c

  return c

@ti.func
def my_cbrt(n):
  # my own cube root function using bisection method, since taichi doesnt have it yet
  iter = 0
  root = 1.0
  if n>1.0:
    a = 0.0
    b = n
    root = (a+b)/2.0
    while (abs(root*root*root-n) > small or iter < 1000):
      root = (a+b)/2.0
      if root*root*root<n:
        a = root
      else:
        b = root
      iter = iter + 1
  elif n<1.0:
    a = 1.0
    b = n
    root = (a+b)/2.0
    while (abs(root*root*root-n) > small or iter < 1000):
      root = (a+b)/2.0
      if root*root*root>n:
        a = root
      else:
        b = root
      iter = iter + 1

  return root


@ti.func
def calc_alpha(c, m):
  # reconstruct interface as line/plane
  # for 3D: use S. Zaleski's Surfer code routine al3d:
  #         find alpha IN: m1 x1 + m2 x2 + m3 x3 = alpha, given m1+m2+m3=1 (all > 0) and VoF value
  #         assumes that cell is unit size, i.e. alpha is relative to dx=dy=dz=1
  #         Note: alpha is not with respect to the lower,front,left corner of the cell. To get it for
  #         this "standard" coordinate system, coordinate mirroring (corrections to alpha) would have to
  #---------------------------------------
  alpha = -small

  if (c > small or c < 1.0-small):
    # convert normal vector into Zaleski's m vector
    mx = ti.abs(m[0])
    my = ti.abs(m[1])
    mz = ti.abs(m[2])

    # the coefficients of the normal must be ordered as: m1 < m2 < m3
    m1 = ti.min(mx,ti.min(my,mz))
    m3 = ti.max(mx,ti.max(my,mz))
    m2 = mx+my+mz-m1-m3

    # get ranges: V1<V2<v3;
    m12 = m1 + m2
    pr  = ti.max(6.0*m1*m2*m3,1.0e-50)
    V1  = m1*m1*m1/pr
    V2  = V1 + 0.5*(m2-m1)/m3
    V3  = 0.0
    mm = 0.0
    if (m3 < m12):
      mm = m3
      V3 = ( m3*m3*(3.0*m12-m3) + m1*m1*(m1-3.0*m3) + m2*m2*(m2-3.0*m3) )/pr
    else:
      mm = m12
      V3 = 0.5*mm/m3

    # limit ch (0.d0 < ch < 0.5d0);
    ch = ti.min(c,1.0-c);

    # calculate d
    if (ch < V1):
      alpha = my_cbrt(pr*ch) # my own cube root function since taichi doesnt have one yet
    elif (ch < V2):
      alpha = 0.5*(m1 + ti.sqrt(m1*m1 + 8.0*m2*m3*(ch-V1)))
    elif (ch < V3):
      p = 2.0*m1*m2
      q = 1.5*m1*m2*(m12 - 2.0*m3*ch)
      p12 = ti.sqrt(p)
      teta = ti.acos(q/(p*p12))/3.0
      cs = ti.cos(teta)
      alpha = p12*(ti.sqrt(3.0*(1.0-cs*cs)) - cs) + m12
    elif (m12 <= m3):
      alpha = m3*ch + 0.5*mm
    else:
      p = m1*(m2+m3) + m2*m3 - 0.25*(m1+m2+m3)*(m1+m2+m3)
      q = 1.5*m1*m2*m3*(0.5-ch)
      p12 = ti.sqrt(p)
      teta = ti.acos(q/(p*p12))/3.0
      cs = ti.cos(teta)
      alpha = p12*(ti.sqrt(3.0*(1.0-cs*cs)) - cs) + 0.5*(m1+m2+m3)

  if (c > 0.5):
    alpha = 1.0-alpha

  return alpha


## data clearing functions ##
@ti.kernel
def clear_data():
  for i,j,k in Flags:
    Flags[i,j,k] = 0
    #C[i,j,k] = 0.0
    M[i,j,k] = [0.0, 0.0, 0.0]
    Phi[i,j,k] = 0.0


@ti.kernel
def clear_data_temp():
  for i,j,k in Flags_temp:
    Flags_temp[i,j,k] = 0
    #C_temp[i,j,k] = 0.0


def clear_data_and_deactivate():
  Flags.ptr.snode().parent.parent.clear_data_and_deactivate()

def clear_data_and_deactivate_temp():
  Flags_temp.ptr.snode().parent.parent.clear_data_and_deactivate()

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
