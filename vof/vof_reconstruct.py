from vof_data import *
from vof_common import *

@ti.kernel
def reconstruct_plic():
  for i,j,k in Flags:
    if is_internal_cell(i,j,k) and is_interface_cell(i,j,k):
      mx,my,mz,alpha = recon(i,j,k)
      M[i,j,k][0] = mx
      M[i,j,k][1] = my
      M[i,j,k][2] = mz
      Alpha[i,j,k] = alpha


@ti.func
def calc_C(alpha, m):
  c = 0.0
  if alpha < 0.0:
    c = 0.0
  elif (alpha > (ti.abs(m[0])+ti.abs(m[1])+ti.abs(m[2]))):
    c = 1.0
  else:
    # convert normal vector into Zaleski's m vector
    a = ti.min(alpha, ti.abs(m[0]) + ti.abs(m[1])+ ti.abs(m[2]) - alpha)
    mx = ti.abs(m[0])
    my = ti.abs(m[1])
    mz = ti.abs(m[2])

    # the coefficients of the normal must be ordered as: m1 < m2 < m3
    m1 = ti.min(mx,my)
    m3 = ti.max(ti.max(mx,my),small)
    m2 = mz*1.0
    if (m2 < m1):
      tmp = m1*1.0
      m1 = m2*1.0
      m2 = tmp*1.0
    elif (m2 > m3):
      tmp = m3*1.0
      m3 = m2*1.0
      m2 = tmp*1.0

    m12 = m1 + m2
    mm  = ti.min(m12,m3)
    pr  = ti.max(6.0*m1*m2*m3,small)
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
def calc_lsq_vof_error(alpha, m, i, j, k):
  error = 0.0
  for dk in range(-1,2):
    for dj in range(-1,2):
      for di in range(-1,2):
        a = alpha - (m[0]*di + m[1]*dj + m[2]*dk)
        err = ti.abs(C[i+di,j+dj,k+dk] - calc_C(a,m))
        error = error + err
  return error


@ti.func
def my_cbrt(n):
  # my own cube root function using bisection method
  iter = 0
  root = 1.0
  if n>1.0:
    a = 0.0
    b = n*1.0
    root = (a+b)/2.0
    while (root*root*root-n >small or iter < 100):
      root = (a+b)/2.0
      if root*root*root<n:
        a = root*1.0
      else:
        b = root*1.0
      iter = iter + 1
  elif n<1.0:
    a = 1.0
    b = n*1.0
    root = (a+b)/2.0
    while (root*root*root-n > small or iter < 100):
      root = (a+b)/2.0
      if root*root*root>n:
        a = root*1.0
      else:
        b = root*1.0
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
  alpha = 0.0

  r13 = 1.0/3.0
  if (c <= Czero or c >= Cone):
    alpha = -1.0e10
  else:
    # convert normal vector into Zaleski's m vector
    mx = ti.abs(m[0])
    my = ti.abs(m[1])
    mz = ti.abs(m[2])

    # the coefficients of the normal must be ordered as: m1 < m2 < m3
    m1 = ti.min(mx,my)
    m3 = ti.max(ti.max(mx,my),small)
    m2 = mz
    if (m2 < m1):
      tmp = m1
      m1 = m2
      m2 = tmp
    elif (m2 > m3):
      tmp = m3
      m3 = m2
      m2 = tmp

    # get ranges: V1<V2<v3;
    m12 = m1 + m2
    pr  = ti.max(6.0*m1*m2*m3,small)
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
      teta = ti.acos(q/(p*p12))*r13
      cs = ti.cos(teta)
      alpha = p12*(ti.sqrt(3.0*(1.0-cs*cs)) - cs) + m12
    elif (m12 <= m3):
      alpha = m3*ch + 0.5*mm
    else:
      p = m1*(m2+m3) + m2*m3 - 0.25*(m1+m2+m3)*(m1+m2+m3)
      q = 1.5*m1*m2*m3*(0.5-ch)
      p12 = ti.sqrt(p)
      teta = ti.acos(q/(p*p12))*r13
      cs = ti.cos(teta)
      alpha = p12*(ti.sqrt(3.0*(1.0-cs*cs)) - cs) + 0.5*(m1+m2+m3)

    if (c > 0.5):
      alpha = (m1+m2+m3)-alpha

  return alpha


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
      h[dj+1][dk+1] = C[i-1,j+dj,k+dk] + C[i,j+dj,k+dk] + C[i+1,j+dj,k+dk]

  # forward, central, backward difference
  hyb = (h[1][1] - h[0][1])
  hyc = (h[2][1] - h[0][1])*.5
  hyf = (h[2][1] - h[1][1])

  hzb = (h[1][1] - h[1][0])
  hzc = (h[1][2] - h[1][0])*.5
  hzf = (h[1][2] - h[1][1])

  # loop over all possible differences
  for kk in range(3): # dont use ti.static(range()) so that loop doesnt unroll, this decreases compiile time
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
        errorMin = error*1.0
        alpha = alp*1.0
        m[0] = n[0]
        m[1] = n[1]
        m[2] = n[2]

  # y-heights
  for dk in ti.static(range(-1,2)):
    for di in ti.static(range(-1,2)):
      h[di+1][dk+1] = C[i+di,j-1,k+dk] + C[i+di,j,k+dk] + C[i+di,j+1,k+dk]

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
        errorMin = error*1.0
        alpha = alp*1.0
        m[0] = n[0]
        m[1] = n[1]
        m[2] = n[2]

  # z-heights
  for dj in ti.static(range(-1,2)):
    for di in ti.static(range(-1,2)):
      h[di+1][dj+1] = C[i+di,j+dj,k-1] + C[i+di,j+dj,k] + C[i+di,j+dj,k+1]

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
        errorMin = error*1.0
        alpha = alp*1.0
        m[0] = n[0]
        m[1] = n[1]
        m[2] = n[2]

  return m[0], m[1], m[2], alpha


# set the reconstruction function
recon = ELVIRA
