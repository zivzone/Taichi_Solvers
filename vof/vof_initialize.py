from vof_data import *
from vof_util import *

@ti.kernel
def init_blocks():
  # loop over all blocks
  for kb in range(nz_tot//b_size):
    for jb in range(ny_tot//b_size):
      for ib in range(nx_tot//b_size):
        x,y,z = get_block_loc(ib,jb,kb)
        phi = get_phi(x,y,z)
        if ti.abs(phi) < b_size*np.sqrt(dx*dx + dy*dy + dz*dz):
          Flags[ib*b_size,jb*b_size,kb*b_size] = 0;

@ti.kernel
def init_cells():
  for i,j,k in Flags:
    x,y,z = get_cell_loc(i,j,k)
    phi = get_phi(x,y,z)
    if ti.abs(phi) <= np.sqrt(dx*dx + dy*dy + dz*dz):
      C[i,j,k] = init_C(x,y,z)
      Flags[i,j,k] = flag_enum.CELL_ACTIVE
    else:
      C[i,j,k] = (1.0+phi/abs(phi))/2.0
      Flags[i,j,k] = flag_enum.CELL_BUFFER

@ti.func
def get_phi_zalesaks_disk(x,y,z):
  # set the initial level set distribution according to Zalesak's notched disk problem
  c = init_radius-ti.sqrt((x-init_center[0])*(x-init_center[0]) + (y-init_center[1])*(y-init_center[1]));
  b1 = init_center[0] - 0.5*init_width;
  b2 = init_center[0] + 0.5*init_width;
  h1 = init_center[1] - init_radius * np.cos(np.arcsin(0.5*init_width/init_radius));
  h2 = init_center[1] - init_radius + init_height;

  phi = 0.0
  if (c >= 0.0 and x <= b1 and y <= h2):
    bb = b1-x
    phi = ti.min(c,bb)
  elif (c >= 0.0 and x >= b2 and y <= h2):
    bb = x-b2
    phi = ti.min(c,bb)
  elif (c >= 0.0 and x >= b1 and x <= b2 and y >= h2):
    bb = y-h2
    phi = ti.min(c,bb)
  elif (c >= 0.0 and x <= b1 and y >= h2):
    bb = ti.sqrt((x-b1)*(x-b1) + (y-h2)*(y-h2))
    phi = ti.min(c,bb)
  elif (c >= 0.0 and x >= b2 and y >= h2):
    bb = ti.sqrt((x-b2)*(x-b2) + (y-h2)*(y-h2))
    phi = ti.min(c,bb)
  elif (x >= b1 and x <= b2 and y <= h2 and y >= h1):
    phi = -ti.min(ti.abs(x-b1),ti.min(ti.abs(x-b2),ti.abs(y-h2)))
  elif (x >= b1 and x <= b2 and y <= h1):
    phi = -ti.min(ti.sqrt((x-b1)*(x-b1)+(y-h1)*(y-h1)), ti.sqrt((x-b2)*(x-b2)+(y-h1)*(y-h1)))
  else:
    phi = c

  return phi


@ti.func
def get_phi_cylinder(x,y,z):
  # set the initial level set distribution to a cylinder
  phi = init_radius \
  - ti.sqrt((x-init_center[0])*(x-init_center[0]) \
  + (y-init_center[1])*(y-init_center[1]))
  return phi


@ti.func
def get_phi_sphere(x,y,z):
  # set the initial level set distribution to a cylinder
  phi = init_radius \
  - ti.sqrt((x-init_center[0])*(x-init_center[0]) \
  + (y-init_center[1])*(y-init_center[1]) \
  + (z-init_center[2])*(z-init_center[2]))
  return phi


@ti.func
def get_phi_plane(x,y,z):
  # set the initial level set distribution to a cylinder
  len = np.sqrt(init_plane_dir[0]**2 + init_plane_dir[1]**2 + init_plane_dir[2]**2)
  a = init_plane_dir[0]/len
  b = init_plane_dir[1]/len
  c = init_plane_dir[2]/len
  d = -(init_center[0]*init_plane_dir[0] + init_center[1]*init_plane_dir[1] + init_center[2]*init_plane_dir[2])/len
  phi = -(a*x + b*y + c*z + d)/ti.sqrt(a*a + b*b + c*c)
  return phi


# set the get phi function
if(init_phi == 0):
  get_phi = get_phi_zalesaks_disk
elif(init_phi == 1):
  get_phi = get_phi_cylinder
elif(init_phi == 2):
  get_phi = get_phi_sphere
else:
  get_phi = get_phi_plane


@ti.func
def init_C(x,y,z):
  # split cell into subcells and estimate volume fraction of each, then sum
  vol = 0.0
  n = n_init_subcells
  dxc = dx/n
  dyc = dy/n
  dzc = dz/n
  for ii in range(n):
    for jj in range(n):
      for kk in range(n):
        xc = x - dx/2.0 + dxc/2.0 + dxc*ii
        yc = y - dy/2.0 + dyc/2.0 + dyc*jj
        zc = z - dz/2.0 + dzc/2.0 + dzc*kk
        phi = get_phi(xc, yc, zc)
        grad_phi = [0.0,0.0,0.0]
        grad_phi[0] = 0.5*(get_phi(xc+dxc, yc, zc)-get_phi(xc-dxc, yc, zc))
        grad_phi[1] = 0.5*(get_phi(xc, yc+dyc, zc)-get_phi(xc, yc-dyc, zc))
        grad_phi[2] = 0.5*(get_phi(xc, yc, zc+dzc)-get_phi(xc, yc, zc-dzc))
        vol = vol + calc_vol_frac_old(phi,grad_phi)/(n*n*n)

  return vol

def initialize():
  init_blocks()
  init_cells()
