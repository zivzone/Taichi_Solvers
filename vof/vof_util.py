from enum import IntFlag, auto
import numpy as np
import matplotlib.pyplot as plt
from vof_data import *

@ti.func
def swap(a,b):
  return b,a

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


## plotting functions ##
def plot_interfaces():
  k = nz_ext
  for j in range(ny_tot):
    for i in range(nx_tot):
      if abs(M[i,j,k][0]) > 100*small or abs(M[i,j,k][1]) > 100*small:
        x,y,z = get_vert_loc(i,j,k)
        if abs(M[i,j,k][0]) > abs(M[i,j,k][1]):
          yl = np.array([y,y+dy])
          if (M[i,j,k][0] > 0.0 and M[i,j,k][1] > 0.0) or (M[i,j,k][0] < 0.0 and M[i,j,k][1] < 0.0):
            yl[0] = max(yl[0],-(M[i,j,k][0]*dx-Alpha[i,j,k])/M[i,j,k][1] + y)
            yl[1] = min(yl[1],-(M[i,j,k][0]*0.0-Alpha[i,j,k])/M[i,j,k][1] + y)
          if (M[i,j,k][0] > 0.0 and M[i,j,k][1] < 0.0) or (M[i,j,k][0] < 0.0 and M[i,j,k][1] > 0.0):
            yl[0] = max(yl[0],-(M[i,j,k][0]*0.0-Alpha[i,j,k])/M[i,j,k][1] + y)
            yl[1] = min(yl[1],-(M[i,j,k][0]*dx-Alpha[i,j,k])/M[i,j,k][1] + y)
          xl = -(M[i,j,k][1]*(yl-y) - Alpha[i,j,k])/M[i,j,k][0] + x
          plt.plot(xl,yl)
        else:
          xl = np.array([x,x+dx])
          if (M[i,j,k][0] > 0.0 and M[i,j,k][1] > 0.0) or (M[i,j,k][0] < 0.0 and M[i,j,k][1] < 0.0):
            xl[0] = max(xl[0],-(M[i,j,k][1]*dy - Alpha[i,j,k])/M[i,j,k][0] + x)
            xl[1] = min(xl[1],-(M[i,j,k][1]*0.0 - Alpha[i,j,k])/M[i,j,k][0] + x)
          if (M[i,j,k][0] > 0.0 and M[i,j,k][1] < 0.0) or (M[i,j,k][0] < 0.0 and M[i,j,k][1] > 0.0):
            xl[0] = max(xl[0],-(M[i,j,k][1]*0.0 - Alpha[i,j,k])/M[i,j,k][0] + x)
            xl[1] = min(xl[1],-(M[i,j,k][1]*dy - Alpha[i,j,k])/M[i,j,k][0] + x)
          yl = -(M[i,j,k][0]*(xl-x) - Alpha[i,j,k])/M[i,j,k][1] + y
          plt.plot(xl,yl)
  plt.grid(color='k', linestyle='-', linewidth=.25)
  plt.xticks(np.arange(0, wx, wx/nx))
  plt.yticks(np.arange(0, wy, wy/ny))
  plt.axis([0, wx, 0, wy])
  plt.show()

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

## level set reconstruction functions ##
@ti.func
def get_phi_from_plic(x,y,z,i,j,k):
  # phi is distance fromm plic plane
  # loc relative to interface cell origin
  x0,y0,z0 = get_vert_loc(i,j,k)
  x = x-x0
  y = y-y0
  z = z-z0

  # phi is distance from plic plane
  phi = -(M[i,j,k][0]*x + M[i,j,k][1]*y + M[i,j,k][2]*z - Alpha[i,j,k]) \
  /ti.sqrt(M[i,j,k][0]*M[i,j,k][0] + M[i,j,k][1]*M[i,j,k][1] + M[i,j,k][2]*M[i,j,k][2])
  return phi


## data clearing functions ##
@ti.kernel
def clear_data():
  for i,j,k in Flags:
    Flags[i,j,k] = 0
    C[i,j,k] = 0.0
    M[i,j,k] = [0.0, 0.0, 0.0]


@ti.kernel
def clear_data_temp():
  for i,j,k in Flags_temp:
    Flags_temp[i,j,k] = 0
    C_temp[i,j,k] = 0.0


def clear_data_and_deactivate():
  Flags.ptr.snode().parent.parent.clear_data_and_deactivate()

def clear_data_and_deactivate_temp():
  Flags_temp.ptr.snode().parent.parent.clear_data_and_deactivate()

## volume fraction estimation functions ##
@ti.func
def sort_four(A):
  low1 = 0.0
  high1 = 0.0
  low2 = 0.0
  high2 = 0.0
  lowest = 0.0
  middle1 = 0.0
  middle2 = 0.0
  highest = 0.0

  if A[0] < A[1]:
    low1 = A[0]
    high1 = A[1]
  else:
    low1 = A[1]
    high1 = A[0]

  if A[2] < A[3]:
    low2 = A[2]
    high2 = A[3]
  else:
    low2 = A[3]
    high2 = A[2]

  if low1 < low2:
    lowest = low1
    middle1 = low2
  else:
    lowest = low2
    middle1 = low1

  if high1 > high2:
    highest = high1
    middle2 = high2
  else:
    highest = high2
    middle2 = high1

  if middle1 < middle2:
    A = ti.Vector([lowest,middle1,middle2,highest])
  else:
    A = ti.Vector([lowest,middle2,middle1,highest])
  return A

@ti.func
def all_sign(phi):
  # check if all vertices are of the same sign
  all_neg = True
  all_pos = True
  for k in ti.static(range(2)):
    for j in ti.static(range(2)):
      for i in ti.static(range(2)):
        if phi[i][j][k] > 0.0:
          all_neg = False
        if phi[i][j][k] < 0.0:
          all_pos = False
  return all_neg, all_pos


@ti.func
def calc_vol_frac_b(phi):
  # compute the volume fraction from level set at vertices using gaussian quadrature

  # set the origin as the vertex with most edges cut by interface
  nmax=0; i0=0; j0=0; k0=0
  for k in ti.static(range(2)):
    for j in ti.static(range(2)):
      for i in ti.static(range(2)):
        n = 0
        if phi[i][j][k]*phi[1-i][j][k] < 0.0: n+=1
        if phi[i][j][k]*phi[i][1-j][k] < 0.0: n+=1
        if phi[i][j][k]*phi[i][j][1-k] < 0.0: n+=1
        if n > nmax: nmax=n; i0=i; j0=j; k0=k

  # swap phis so that origin is at 0,0,0
  if i0 == 1:
    for k in ti.static(range(2)):
      for j in ti.static(range(2)):
        temp = phi[0][j][k]
        phi[0][j][k] = phi[1][j][k]
        phi[1][j][k] = temp
  if j0 == 1:
    for k in ti.static(range(2)):
      for i in ti.static(range(2)):
        temp = phi[i][0][k]
        phi[i][0][k] = phi[i][1][k]
        phi[i][1][k] = temp
  if k0 == 1:
    for j in ti.static(range(2)):
      for i in ti.static(range(2)):
        temp = phi[i][j][0]
        phi[i][j][0] = phi[i][j][1]
        phi[i][j][1] = temp

  # get distance from the vertex to intersection point on each edge
  l = [1.0,1.0,1.0]
  if phi[0][0][0]*phi[1][0][0] < 0:
    l[0] = -phi[0][0][0]/(phi[1][0][0]-phi[0][0][0])
  if phi[0][0][0]*phi[0][1][0] < 0:
    l[1] = -phi[0][0][0]/(phi[0][1][0]-phi[0][0][0])
  if phi[0][0][0]*phi[0][0][1] < 0:
    l[2] = -phi[0][0][0]/(phi[0][0][1]-phi[0][0][0])

  # polynomial coefficients
  B = phi[0][0][0]
  Bx = phi[1][0][0]-phi[0][0][0]
  By = phi[0][1][0]-phi[0][0][0]
  Bz = phi[0][0][1]-phi[0][0][0]
  Bxy = phi[1][1][0]-phi[1][0][0]-phi[0][1][0]+phi[0][0][0]
  Byz = phi[0][1][1]-phi[0][1][0]-phi[0][0][1]+phi[0][0][0]
  Bxz = phi[1][0][1]-phi[1][0][0]-phi[0][0][1]+phi[0][0][0]
  Bxyz = phi[1][1][1]-phi[1][1][0]-phi[1][0][1]+phi[1][0][0] \
        -phi[0][1][1]+phi[0][1][0]+phi[0][0][1]-phi[0][0][0]

  # choose the integration order by sorting the distances, max to min
  # swap data
  order = ti.Vector([0,1,2])
  if l[0] < l[1]:
    order[0],order[1] = swap(order[0],order[1])
    l[0],l[1] = swap(l[0],l[1])
    Bx,By = swap(Bx,By)
    Bxz,Byz = swap(Bxz,Byz)
  if l[1] < l[2]:
    order[1],order[2] = swap(order[2],order[1])
    l[1],l[2] = swap(l[1],l[2])
    By,Bz = swap(By,Bz)
    Bxy,Bxz = swap(Bxy,Bxz)
  if l[0] < l[1]:
    order[0],order[1] = swap(order[0],order[1])
    l[0],l[1] = swap(l[0],l[1])
    Bx,By = swap(Bx,By)
    Bxz,Byz = swap(Bxz,Byz)

  # 3 point 2d gaussian quadrature of z
  xq = [-np.sqrt(3.0/5.0), 0, np.sqrt(3.0/5.0)]; # quadrature points
  wq = [5.0/9.0, 8.0/9.0, 5.0/9.0];              # quadrature weights

  vf = 0.0
  Jx = l[0]/2.0 # jacobian
  z0 = 0.0
  for i in ti.static(range(3)):
    x = (xq[i]+1.0)*Jx
    # y integration bounds depends on x
    y1 = -((Bxz*z0 + Bx)*x + Bz*z0 + B) / ((Bxyz*z0 + Bxy)*x + Byz*z0 + By + small)
    if y1 < 0.0 or y1 > 1.0:
      # when there is no sensical y bound
      y1 = 1.0
    Jy = y1/2.0
    for j in ti.static(range(3)):
      y = (xq[j]+1.0)*Jy
      z = -((Bxy*y + Bx)*x + By*y + B) / ((Bxyz*y + Bxz)*x + Byz*y + Bz)  # z location of interface
      vf+= z*wq[i]*wq[j]*Jx*Jy

  if phi[0][0][0] < 0.0:
    vf = 1.0-vf

  return vf


@ti.func
def calc_vol_frac(phi):
  # compute the volume fraction from level set at vertices
  # by splitting the cell into elementary cases then using gaussian quadrature
  vf = 0.0

  all_neg,all_pos = all_sign(phi)
  if not all_pos and not all_neg:
    # count number and store location of cut edges in each direction
    ni = 0
    li = ti.Vector([1.0,1.0,1.0,1.0])
    for k in ti.static(range(2)):
      for j in ti.static(range(2)):
        if phi[0][j][k]*phi[1][j][k] < 0.0:
          li[j+2*k] = -phi[0][j][k]/(phi[1][j][k]-phi[0][j][k])
          ni+=1
    nj = 0
    lj = ti.Vector([1.0,1.0,1.0,1.0])
    for k in ti.static(range(2)):
      for i in ti.static(range(2)):
        if phi[i][0][k]*phi[i][1][k] < 0.0:
          lj[i+2*k] = -phi[i][0][k]/(phi[i][1][k]-phi[i][0][k])
          nj+=1
    nk = 0
    lk = ti.Vector([1.0,1.0,1.0,1.0])
    for j in ti.static(range(2)):
      for i in ti.static(range(2)):
        if phi[i][j][0]*phi[i][j][1] < 0.0:
          lk[i+2*j] = -phi[i][j][0]/(phi[i][j][1]-phi[i][j][0])
          nk+=1

    # choose the direction with the least number of cuts
    nd = 0
    dir = 1
    l = ti.Vector([1.0,1.0,1.0,1.0])
    if ni <= nj and ni <= nk:
      dir = 0
      nd = ni
      l = li
    if nj <= ni and nj <= nk:
      dir = 1
      nd = nj
      l = lj
    if nk <= ni and nk <= nj:
      dir = 2
      nd = nk
      l = lk

    # rotate the cell so that x-axis is the chosen direction
    phi_temp = [[[0.0,0.0],[0.0,0.0]],
                [[0.0,0.0],[0.0,0.0]]]
    for k in ti.static(range(2)):
      for j in ti.static(range(2)):
        for i in ti.static(range(2)):
          phi_temp[i][j][k] = phi[i][j][k]
    for k in ti.static(range(2)):
      for j in ti.static(range(2)):
        for i in ti.static(range(2)):
          if dir == 1:
            phi[i][j][k] = phi_temp[j][i][k]
          elif dir == 2:
            phi[i][j][k] = phi_temp[k][j][i]
    for k in ti.static(range(2)):
      for j in ti.static(range(2)):
        for i in ti.static(range(2)):
          phi_temp[i][j][k] = phi[i][j][k]

    # sort intersection locations, there should be a max of two for the alorithm to work properly
    l = sort_four(l)

    # calculate volume fraction of subcells
    l_old = 0.0
    for n in ti.static(range(4)):
      # interpolate phi along edge at cut locations to get subcell vertex phi
      for k in ti.static(range(2)):
        for j in ti.static(range(2)):
          phi[1][j][k] = phi_temp[0][j][k] + l[n]*(phi_temp[1][j][k]-phi_temp[0][j][k])

      all_neg,all_pos = all_sign(phi)
      if not all_pos and not all_neg and (l[n]-l_old) > 0.0:
        vf += calc_vol_frac_b(phi)*(l[n]-l_old)
      elif all_pos:
        vf += l[n]-l_old

      l_old = l[n]

      #start next subcell at end of the last subcell
      for k in ti.static(range(2)):
        for j in ti.static(range(2)):
          phi[0][j][k] = phi_temp[0][j][k] + l[n]*(phi_temp[1][j][k]-phi_temp[0][j][k])

  elif all_pos:
    vf = 1.0

  return vf

@ti.func
def calc_vol_frac_old(phi,grad_phi):
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
  vol = 0.0
  if dxi > Czero :
    if deta > Czero :
      if dzeta > Czero : #  3D
        vol = (a*a*a - b*b*b - c*c*c - d*d*d + e*e*e)/(6.0*dxi*deta*dzeta)
      else:#  2D
        vol = (a*a - c*c)/(2.0*dxi*deta)
    else: #  1D
      vol = a/dxi
  else:  #  0D
    if phim !=  0.0:
      vol = 0.0
    else:
      vol = 0.5

  if phi > 0.0 :
    vol = 1.0 - vol

  return vol

## cell identifier functions
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
