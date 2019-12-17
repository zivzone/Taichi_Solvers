from vof_data import *
from vof_common import *

@ti.kernel
def interp_face_velocity_to_vertex():
  # interpolate face center volocity components to cell vertices
  for i,j,k in Flags:
    if is_internal_vertex(i,j,k):
      Vel_vert[i,j,k][0] = (U[i,j,k] + U[i-1,j,k])/2.0
      Vel_vert[i,j,k][1] = (V[i,j,k] + V[i,j-1,k])/2.0
      Vel_vert[i,j,k][2] = (W[i,j,k] + W[i,j,k-1])/2.0


@ti.kernel
def back_track_DMC(dt: ti.f32):
  # compute the Dual Mesh Characteristic velocity at vertices
  for i,j,k in Flags:
    if is_internal_vertex(i,j,k):
      x,y,z = get_vert_loc(i,j,k):
      # x-direction
      a = 0.0
      if Vel_vert[i,j,k][0] <=0:
        a = (Vel_vert[i,j,k][0] - Vel_vert[i-1,j,k][0])/dx
      else:
        a = -(Vel_vert[i,j,k][0] - Vel_vert[i+1,j,k][0])/dx

      if ti.abs(a) < small:
        Vert_pos[i,j,k][0] = x - dt*Vel_vert[i,j,k][0]
      else:
        Vert_pos[i,j,k][0] = x - (1.0-ti.exp(-a*dt))*Vel_vert[i,j,k][0]/a

      # y-direction
      if Vel_vert[i,j,k][1] <=0:
        a = (Vel_vert[i,j,k][1] - Vel_vert[i,j-1,k][1])/dy
      else:
        a = -(Vel_vert[i,j,k][1] - Vel_vert[i,j+1,k][1])/dy

      if ti.abs(a) < small:
        Vert_pos[i,j,k][1] = y - dt*Vel_vert[i,j,k][0]
      else:
        Vert_pos[i,j,k][1] = y - (1.0-ti.exp(-a*dt))*Vel_vert[i,j,k][1]/a

      # z-direction
      if Vel_vert[i,j,k][2] <=0:
        a = (Vel_vert[i,j,k][2] - Vel_vert[i,j,k-1][2])/dz
      else:
        a = -(Vel_vert[i,j,k][2] - Vel_vert[i,j,k+1][2])/dz

      if ti.abs(a) < small:
        Vert_pos[i,j,k][2] = z - dt*Vel_vert[i,j,k][2]
      else:
        Vert_pos[i,j,k][2] = z - (1.0-ti.exp(-a*dt))*Vel_vert[i,j,k][2]/a

@ti.kernel
def compute_DC(dt: ti.f32):
  # compute volume fraction fluxes using an isoadvector-like algorithm,
  # ie. the flux is the "time integral of the submerged face area"
  for i,j,k in Flags:
    # flux the left face
    if is_internal_x_face(i,j,k) and is_active_x_face(i,j,k):
      # find the "upwind" interface cell
      iuw,juw,kuw = get_upwind_x(i,j,k)

      # intialize DCx to upwind volume fraction
      DCx[i,j,k] = C[iuw,juw,kuw]
      if is_interface_cell(iuw,juw,kuw):
        # compute the level set at each vertex on this face at time t
        phi = [[[0.0,0.0],[0.0,0.0]],
               [[0.0,0.0],[0.0,0.0]]] # 3d array (y,z,t)
        x,y,z = get_vert_loc(i,j,k);
        phi[0][0][0] = get_phi_from_plic(x,y,z,iuw,juw,kuw)
        phi[1][0][0] = get_phi_from_plic(x,y+dy,z,iuw,juw,kuw)
        phi[0][1][0] = get_phi_from_plic(x,y,z+dz,iuw,juw,kuw)
        phi[1][1][0] = get_phi_from_plic(x,y+dy,z+dz,iuw,juw,kuw)

        # lagrangian backtrack the vertices using DMC to compute the level set at time t+dt
        x_dmc,y_dmc,z_dmc = backtrack_dmc(x,y,z,i,j,k,dt)
        phi[0][0][1] = get_phi_from_plic(Pos_vert[i,j,k][0],Pos_vert[i,j,k][1],Pos_vert[i,j,k][2],iuw,juw,kuw)

        x_dmc,y_dmc,z_dmc = backtrack_dmc(x,y+dx,z,i,j+1,k,dt)
        phi[1][0][1] = get_phi_from_plic(Pos_vert[i,j+1,k][0],Pos_vert[i,j+1,k][1],Pos_vert[i,j+1,k][2],iuw,juw,kuw)

        x_dmc,y_dmc,z_dmc = backtrack_dmc(x,y,z+dz,i,j,k+1,dt)
        phi[0][1][1] = get_phi_from_plic(Pos_vert[i,j,k+1][0],Pos_vert[i,j,k+1][1],Pos_vert[i,j,k+1][2],iuw,juw,kuw)

        x_dmc,y_dmc,z_dmc = backtrack_dmc(x,y+dy,z+dz,i,j+1,k+1,dt)
        phi[1][1][1] = get_phi_from_plic(Pos_vert[i,j+1,k+1][0],Pos_vert[i,j+1,k+1][1],Pos_vert[i,j+1,k+1][2],iuw,juw,kuw)

        #vol = calc_vol_frac(phi, dy, dz, dt)

@ti.func
def get_upwind_x(i,j,k):
  # get the "upwind" interface cell of this face
  # try the face neighbors first
  iup = i
  jup = j
  kup = k
  sgn = -1.0
  if U[i,j,k] > 0.0:
    iup = i-1
    sgn = 1.0

  if not is_interface_cell(iup,j,k):
    # the face neigbor is not an interface cell
    # instead choose the edge or vertex neigbor with the highest upwind velocity
    for dk in ti.static(range(-1,2)):
      for dj in ti.static(range(-1,2)):
        umax = 0.0
        if is_interface_cell(iup,j+dj,k+dk) and sgn*U[iup,j+dj,k+dk] > umax:
          jup = j+dj
          kup = k+dk
          umax = sgn*U[iup,j+dj,k+dk]

  return iup,jup,kup

@ti.func
def get_phi_from_plic(x,y,z,i,j,k):
  # loc relative to interface cell origin
  x0,y0,z0 = get_vert_loc(i,j,k)
  x = x-x0
  y = y-y0
  z = z-z0

  # phi is distance from plic plane
  phi = (M[i,j,k][0]*x + M[i,j,k][1]*y + M[i,j,k][2]*z - Alpha[i,j,k])/np.sqrt(3.0)
  return phi

@ti.func
def backtrack_dmc(x,y,z,i,j,k,dt):
  x_dmc = x-Vel_vert_dmc[i,j,k][0]*dt
  y_dmc = y-Vel_vert_dmc[i,j,k][1]*dt
  z_dmc = z-Vel_vert_dmc[i,j,k][2]*dt
  return x_dmc, y_dmc, z_dmc

@ti.func
def sort_four(A):
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
    A = [lowest,middle1,middle2,highest]
  else:
    A = [lowest,middle2,middle1,highest]
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
  eps = 1.0e-20
  # compute the volume fraction from level set at vertices using gaussian quadrature

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

  # set the origin as the vertex with most edges cut by interface
  nmax=0; ijk0 = [0,0,0]
  for k in ti.static(range(2)):
    for j in ti.static(range(2)):
      for i in ti.static(range(2)):
        n = 0
        if phi[i][j][k]*phi[1-i][j][k] < 0.0: n+=1
        if phi[i][j][k]*phi[i][1-j][k] < 0.0: n+=1
        if phi[i][j][k]*phi[i][j][1-k] < 0.0: n+=1
        if n > nmax: nmax=n; ijk0[0]=i; ijk0[1]=j; ijk0[2]=k

  # get distance from the vertex to intersection point on each edge
  l = [1.0,1.0,1.0]
  i0 = ijk0[0]; j0 = ijk0[1]; k0 = ijk0[2];
  if phi[i0][j0][k0]*phi[1-i0][j0][k0] < 0:
    l[0] = -phi[i0][j0][k0]/(phi[1-i0][j0][k0]-phi[i0][j0][k0])
  if phi[i0][j0][k0]*phi[i0][1-j0][k0] < 0:
    l[1] = -phi[i0][j0][k0]/(phi[i0][1-j0][k0]-phi[i0][j0][k0])
  if phi[i0][j0][k0]*phi[i0][j0][1-k0] < 0:
    l[2] = -phi[i0][j0][k0]/(phi[i0][j0][1-k0]-phi[i0][j0][k0])

  # choose the integration order by sorting the distances, max to min
  # swap data
  order = [0,1,2]
  if l[0] < l[1]:
    order[0],order[1] = order[1],order[0]
    l[0],l[1] = l[1],l[0]
    ijk0[0],ijk0[1] = ijk0[1],ijk0[0]
    Bx,By = By,Bx
    Bxz,Byz = Byz,Bxz
  if l[1] < l[2]:
    order[1],order[2] = order[2],order[1]
    l[1],l[2] = l[2],l[1]
    ijk0[1],ijk0[2] = ijk0[2],ijk0[1]
    Bz,By = By,Bz
    Bxy,Bxz = Bxz,Bxy
  if l[0] < l[1]:
    order[0],order[1] = order[1],order[0]
    l[0],l[1] = l[1],l[0]
    ijk0[0],ijk0[1] = ijk0[1],ijk0[0]
    Bx,By = By,Bx
    Bxz,Byz = Byz,Bxz

  # 2d gaussian quadrature of height
  n_qpts = 3
  xq = [-np.sqrt(3.0/5.0), 0, np.sqrt(3.0/5.0)]; # quadrature points
  wq = [5.0/9.0, 8.0/9.0, 5.0/9.0];              # quadrature weights

  vol = 0.0
  xl = l[0]*(1.0-ijk0[0]) + (1.0-l[0])*ijk0[0] # l as x location instead of distance from origin
  x0 = min(xl,ijk0[0])
  x1 = max(xl,ijk0[0])
  Jx = (x1-x0)/2.0 # jacobian
  z0 = ijk0[2]
  for i in range(n_qpts):
    x = (xq[i]+1.0)*Jx + x0
    # y integration bounds depends on x
    y01 = -((Bxz*z0 + Bx)*x + Bz*z0 + B) / ((Bxyz*z0 + Bxy)*x + Byz*z0 + By + eps)
    y01 = max(min(y01,1.0),0.0)
    if ijk0[1] == 1:
      y0 = y01
      y1 = 1.0
    else:
      y0 = 0.0
      y1 = y01
    Jy = (y1-y0)/2.0
    for j in ti.static(range(n_qpts)):
      y = (xq[j]+1.0)*Jy + y0
      z = -((Bxy*y + Bx)*x + By*y + B) / ((Bxyz*y + Bxz)*x + Byz*y + Bz)  # z location of interface
      z = (1.0-ijk0[2])*z + (1.0-z)*ijk0[2]      # turn z location into height

      vol+= z*wq[i]*wq[j]*Jx*Jy

  if phi[i0][j0][k0] < 0.0:
    vol = 1.0-vol

  return vol

@ti.func
def calc_vol_frac(phi):
  # compute the volume fraction from level set at vertices
  # by splitting the cell into elementary cases then using gaussian quadrature
  vol = 0.0

  all_neg,all_pos = all_sign(phi)
  if not all_pos and not all_neg:
    # count number and store location of cut edges in each direction
    ni = 0
    li = [1.0,1.0,1.0,1.0]
    for k in ti.static(range(2):
      for j in ti.static(range(2):
        if phi[0][j][k]*phi[1][j][k] < 0:
          li[ni] = -phi[0][j][k]/(phi[1][j][k]-phi[0][j][k])
          ni+=1
    nj = 0
    lj = [1.0,1.0,1.0,1.0]
    for k in ti.static(range(2)):
      for i in ti.static(range(2)):
        if phi[i][0][k]*phi[i][1][k] < 0:
          lj[nj] = -phi[i][0][k]/(phi[i][1][k]-phi[i][0][k])
          nj+=1
    nk = 0
    lk = [1.0,1.0,1.0,1.0]
    for j in ti.static(range(2)):
      for i in ti.static(range(2)):
        if phi[i][j][0]*phi[i][j][1] < 0:
          lk[nk] = -phi[i][j][0]/(phi[i][j][1]-phi[i][j][0])
          nk+=1

    # choose the direction with the least number of cuts
    nd = 0
    if ni <= nj and ni <= nk: dir = 0; nd = ni; l = li
    if nj <= ni and nj <= nk: dir = 1; nd = nj; l = lj
    if nk <= ni and nk <= nj: dir = 2; nd = nk; l = lk

    # rotate the cell so that x-copy()axis is the chosen direction
    phi_temp = phi # assign value not reference
    for k in ti.static(range(2)):
      for j in ti.static(range(2)):
        for i in ti.static(range(2)):
          if dir == 1:
            phi[i][j][k] = phi_temp[j][i][k]
          elif dir == 2:
            phi[i][j][k] = phi_temp[k][j][i]
    phi_temp = phi

    # sort intersection locations, there should be a max of two for the alorithm to work properly
    l = sort_four(l)

    # calculate volume fraction of subcells
    l_old = 0.0
    for n in range(nd+1):
      # interpolate phi along edge at cut locations to get subcell vertex phi
      for k in ti.static(range(2):
        for j in ti.static(range(2):
          phi[1][j][k] = phi_temp[0][j][k] + l[n]*(phi_temp[1][j][k]-phi_temp[0][j][k])

      all_neg,all_pos = all_sign(phi)
      if not all_pos and not all_neg:
        vol += calc_vol_frac_b(phi)*(l[n]-l_old)
      elif all_pos:
        vol += l[n]-l_old

      l_old = l[n]

      #start next subcell at end of the last subcell
      for k in ti.static(range(2)):
        for j in ti.static(range(2)):
          phi[0][j][k] = phi[1][j][k]

  elif all_pos:
    vol = 1.0

  return vol
