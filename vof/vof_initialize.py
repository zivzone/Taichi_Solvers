from vof_data import *

@ti.kernel
def initialize():
  # loop over all blocks
  for ib in range(n_x//b_size):
    for jb in range(n_y//b_size):
      for kb in range(n_z//b_size):
        x,y,z = get_block_loc(ib,jb,kb)
        phi = get_phi(x,y,z)
        if ti.abs(phi) < b_size*np.sqrt(dx*dx + dy*dy + dz*dz):
          # if the block is near interface
          # loop over cells in block
          for ic in range(b_size):
            for jc in range(b_size):
              for kc in range(b_size):
                i = ic + ib*b_size
                j = jc + jb*b_size
                k = kc + kb*b_size
                x,y,z = get_cell_loc(i,j,k)
                phi = get_phi(x,y,z)
                if ti.abs(phi) < np.sqrt(dx*dx + dy*dy + dz*dz):
                  C[i,j,k] = init_C(x,y,z)
                  Flags[i,j,k] = flag_enum.CELL_ACTIVE
                else:
                  C[i,j,k] = (1.0+phi/ti.abs(phi))/2.0
                  Flags[i,j,k] = flag_enum.CELL_GHOST


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


## get_phi functions ##
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
  phi = (a*x + b*y + c*z + d)/ti.sqrt(a*a + b*b + c*c)
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
def get_block_loc(ib,jb,kb):
  x = ib*dx*b_size + dx*b_size/2.0 - n_ghost*dx
  y = jb*dy*b_size + dy*b_size/2.0 - n_ghost*dy
  z = kb*dz*b_size + dz*b_size/2.0 - n_ghost*dz
  return x,y,z


@ti.func
def get_cell_loc(i,j,k):
  x = i*dx + dx/2.0 - n_ghost*dx
  y = j*dy + dy/2.0 - n_ghost*dy
  z = k*dz + dz/2.0 - n_ghost*dz
  return x,y,z


@ti.func
def estimate_C(phi,grad_phi):
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
        vol = vol + estimate_C(phi,grad_phi)/(n*n*n)

  return vol
