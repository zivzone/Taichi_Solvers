from vof_data import *
from vof_util import *

@ti.kernel
def reconstruct_plic():
  for i,j,k in Flags:
    if is_internal_cell(i,j,k) and is_interface_cell(i,j,k):
      alpha,m = recon(i,j,k)
      #transform normal vector and alpha into physical space
      Alpha[i,j,k] = alpha + min(0.0,m[0]) + min(0.0,m[1]) + min(0.0,m[2])
      M[i,j,k][0] = m[0]/dx
      M[i,j,k][1] = m[1]/dy
      M[i,j,k][2] = m[2]/dz


@ti.kernel
def reconstruct_phi_from_plic():
  # reconstruct level set in cells near the interface as the weighted average
  # of the signed distances from plic planes in nearby cells
  for i,j,k in Flags:
    if is_active_cell(i,j,k) or is_buffer_cell(i,j,k):
      num = small
      den = small
      x,y,z = get_cell_loc(i,j,k)
      for di,dj,dk in ti.ndrange((-1,2),(-1,2),(-1,2)):
        if is_interface_cell(i+di,j+dj,k+dk):
          phi,w = get_phi_and_weight_from_plic(x,y,z,i+di,j+dj,k+dk)
          num += phi*w
          den += w
      Phi[i,j,k] = num/den

@ti.kernel
def reconstruct_plic_from_phi():
  for i,j,k in Flags:
    if is_internal_cell(i,j,k) and is_interface_cell(i,j,k):
      m = ti.Vector([0.0,0.0,0.0])
      m[0] = (Phi[i+1,j,k]-Phi[i-1,j,k])/(2.0)
      m[1] = (Phi[i,j+1,k]-Phi[i,j-1,k])/(2.0)
      m[2] = (Phi[i,j,k+1]-Phi[i,j,k-1])/(2.0)
      m = -m/(ti.abs(m[0]) + ti.abs(m[1]) + ti.abs(m[2]))
      alpha = calc_alpha(C[i,j,k], m)

      #transform normal vector and alpha into physical space
      Alpha[i,j,k] = alpha + min(0.0,m[0]) + min(0.0,m[1]) + min(0.0,m[2])
      M[i,j,k][0] = m[0]/dx
      M[i,j,k][1] = m[1]/dy
      M[i,j,k][2] = m[2]/dz


@ti.func
def calc_lsq_vof_error(alpha, m, i, j, k):
  error = 0.0
  for dk in range(-1,2):
    for dj in range(-1,2):
      for di in range(-1,2):
        a = alpha - (m[0]*di + m[1]*dj + m[2]*dk)
        err = ti.abs(C[i+di,j+dj,k+dk] - calc_C(a,m))
        error = error + err*err
  return error


@ti.func
def ELVIRA(i, j, k):
  # reconstruct planar interface using ELVIRA
  # check all possible normal vectors using forward, central, and backward differrences
  h = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
  n = [0.0,0.0,0.0]
  m = [0.0,0.0,0.0]
  alpha = 0.0

  errorMin = 1.0e20

  # x-heights
  for dk in ti.static(range(-1,2)):
    for dj in ti.static(range(-1,2)):
      h[dj+1][dk+1] = 0.0
      for di in ti.static(range(-1,1+1)):
        h[dj+1][dk+1] += C[i+di,j+dj,k+dk]

  # forward, central, backward difference
  hyb = (h[1][1] - h[0][1])
  hyc = (h[2][1] - h[0][1])*.5
  hyf = (h[2][1] - h[1][1])

  hzb = (h[1][1] - h[1][0])
  hzc = (h[1][2] - h[1][0])*.5
  hzf = (h[1][2] - h[1][1])

  # loop over all possible differences
  for kk in range(3): # dont use ti.static(range()) so that loop doesnt unroll
    for jj in range(3):
      if jj == 0:
        n[1] = hyb
      elif jj== 1:
        n[1] = hyc
      else:
        n[1] = hyf

      if kk == 0:
        n[2] = hzb
      elif kk== 1:
        n[2] = hzc
      else:
        n[2] = hzf

      n[0] = 1.0
      if ((C[i+1,j,k] - C[i-1,j,k]) < 0.0):
        n[0] = -1.0

      # make sum of components = 1 for PLIC reconstruction and reconstruct
      rdenom = 1.0/(ti.abs(n[0]) + ti.abs(n[1]) + ti.abs(n[2]))
      n[0] = -n[0]*rdenom
      n[1] = -n[1]*rdenom
      n[2] = -n[2]*rdenom
      alp =  calc_alpha(C[i,j,k], n)
      error = calc_lsq_vof_error(alp,n,i,j,k)

      if (error < errorMin):
        errorMin = error
        alpha = alp
        m[0] = n[0]
        m[1] = n[1]
        m[2] = n[2]

  # y-heights
  for dk in ti.static(range(-1,2)):
    for di in ti.static(range(-1,2)):
      h[di+1][dk+1] = 0.0
      for dj in ti.static(range(-1,1+1)):
        h[di+1][dk+1] += C[i+di,j+dj,k+dk]

  # forward, central, backward differences
  hxb = (h[1][1] - h[0][1])
  hxc = (h[2][1] - h[0][1])*.5
  hxf = (h[2][1] - h[1][1])

  hzb = (h[1][1] - h[1][0])
  hzc = (h[1][2] - h[1][0])*.5
  hzf = (h[1][2] - h[1][1])

  # loop over all possible differences
  for kk in range(3):
    for ii in range(3):
      if ii == 0:
        n[0] = hxb
      elif ii== 1:
        n[0] = hxc
      else:
        n[0] = hxf

      if kk == 0:
        n[2] = hzb
      elif kk==1:
        n[2] = hzc
      else:
        n[2] = hzf

      n[1] = 1.0
      if ((C[i,j+1,k] - C[i,j-1,k]) < 0.0):
        n[1] = -1.0

      # make sum of components = 1 for PLIC reconstruction and reconstruct
      rdenom = 1.0/(ti.abs(n[0]) + ti.abs(n[1]) + ti.abs(n[2]))
      n[0] = -n[0]*rdenom
      n[1] = -n[1]*rdenom
      n[2] = -n[2]*rdenom
      alp = calc_alpha(C[i,j,k], n)
      error = calc_lsq_vof_error(alp,n,i,j,k)

      if (error < errorMin):
        errorMin = error
        alpha = alp
        m[0] = n[0]
        m[1] = n[1]
        m[2] = n[2]

  # z-heights
  for dj in ti.static(range(-1,2)):
    for di in ti.static(range(-1,2)):
      h[di+1][dj+1] = 0.0
      for dk in ti.static(range(-1,1+1)):
        h[di+1][dj+1] += C[i+di,j+dj,k+dk]

  # forward, central, backward differences
  hxb = (h[1][1] - h[0][1])
  hxc = (h[2][1] - h[0][1])*.5
  hxf = (h[2][1] - h[1][1])

  hyb = (h[1][1] - h[1][0])
  hyc = (h[1][2] - h[1][0])*.5
  hyf = (h[1][2] - h[1][1])

  # loop over all possible differences
  for jj in range(3):
    for ii in range(3):
      if ii == 0:
        n[0] = hxb
      elif ii== 1:
        n[0] = hxc
      else:
        n[0] = hxf

      if jj == 0:
        n[1] = hyb
      elif jj==1:
        n[1] = hyc
      else:
        n[1] = hyf

      n[2] = 1.0
      if ((C[i,j,k+1] - C[i,j,k-1]) < 0.0):
        n[2] = -1.0

      # make sum of components = 1 for PLIC reconstruction and reconstruct
      rdenom = 1.0/(ti.abs(n[0]) + ti.abs(n[1]) + ti.abs(n[2]));
      n[0] = -n[0]*rdenom
      n[1] = -n[1]*rdenom
      n[2] = -n[2]*rdenom
      alp = calc_alpha(C[i,j,k], n)
      error = calc_lsq_vof_error(alp,n,i,j,k)

      if (error < errorMin):
        errorMin = error
        alpha = alp
        m[0] = n[0]
        m[1] = n[1]
        m[2] = n[2]

  return alpha,m


@ti.func
def Young(i,j,k):
  m = ti.Vector([0.0,0.0,0.0])
  # average the gradient computed at eight vertices
  for di,dj,dk in ti.static(ti.ndrange((0,2),(0,2),(0,2))):
    m[0] += ((C[i+di,j+dj,k+dk]+C[i+di,j+dj-1,k+dk]+C[i+di,j+dj,k+dk-1]+C[i+di,j+dj-1,k+dk-1]) \
           -(C[i+di-1,j+dj,k+dk]+C[i+di-1,j+dj-1,k+dk]+C[i+di-1,j+dj,k+dk-1]+C[i+di-1,j+dj-1,k+dk-1]))/4.0
    m[1] += ((C[i+di,j+dj,k+dk]+C[i+di-1,j+dj,k+dk]+C[i+di,j+dj,k+dk-1]+C[i+di-1,j+dj,k+dk-1]) \
           -(C[i+di,j+dj-1,k+dk]+C[i+di-1,j+dj-1,k+dk]+C[i+di,j+dj-1,k+dk-1]+C[i+di-1,j+dj-1,k+dk-1]))/4.0
    m[2] += ((C[i+di,j+dj,k+dk]+C[i+di-1,j+dj,k+dk]+C[i+di,j+dj-1,k+dk]+C[i+di-1,j+dj-1,k+dk]) \
           -(C[i+di,j+dj,k+dk-1]+C[i+di-1,j+dj,k+dk-1]+C[i+di,j+dj-1,k+dk-1]+C[i+di-1,j+dj-1,k+dk-1]))/4.0

  m = m/8.0
  len = ti.abs(m[0]) + ti.abs(m[1]) + ti.abs(m[2])
  if len < small:
    len = 1.0
    m[0] = 1.0
    m[1] = 0.0
    m[2] = 0.0
  m = -m/len
  alpha = calc_alpha(C[i,j,k], m)
  return alpha,m


# set the reconstruction function
recon = ELVIRA


@ti.kernel
def check_vof():
  ti.serialize()
  for i,j,k in Flags:
    if is_internal_cell(i,j,k):
      Tot_vol[None] = Tot_vol[None] + C[i,j,k]*vol
    """
    if is_interface_cell(i,j,k):
      #calculate the volume fraction reconstructed from phi
      phi = [[[0.0,0.0],[0.0,0.0]],
             [[0.0,0.0],[0.0,0.0]]] # 3d array (y,z,t)
      x,y,z = get_vert_loc(i,j,k);
      phi[0][0][0] = get_phi_from_plic(x,y,z,i,j,k)
      phi[1][0][0] = get_phi_from_plic(x+dx,y,z,i,j,k)
      phi[0][1][0] = get_phi_from_plic(x,y+dy,z,i,j,k)
      phi[1][1][0] = get_phi_from_plic(x+dx,y+dy,z,i,j,k)
      phi[0][0][1] = get_phi_from_plic(x,y,z+dz,i,j,k)
      phi[1][0][1] = get_phi_from_plic(x+dx,y,z+dz,i,j,k)
      phi[0][1][1] = get_phi_from_plic(x,y+dy,z+dz,i,j,k)
      phi[1][1][1] = get_phi_from_plic(x+dx,y+dy,z+dz,i,j,k)

      alpha,m = calc_plic_from_phi(phi)
      vf = calc_C(alpha,m)

      if abs(C[i,j,k] - vf) > 1.0e-10:
        print(vf)
        print(C[i,j,k])
        print(m[0])
        print(m[1])
        print(m[2])

    """
