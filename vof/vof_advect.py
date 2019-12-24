from vof_data import *
from vof_common import *

@ti.kernel
def interp_face_velocity_to_vertex():
  # interpolate face center volocity components to cell vertices
  for i,j,k in Flags:
    if is_internal_vertex(i,j,k) or is_ghost_vertex(i,j,k):
      Vel_vert[i,j,k][0] = (U[i,j,k] + U[i-1,j,k])/2.0
      Vel_vert[i,j,k][1] = (V[i,j,k] + V[i,j-1,k])/2.0
      Vel_vert[i,j,k][2] = (W[i,j,k] + W[i,j,k-1])/2.0


@ti.kernel
def back_track_DMC():
  # compute the Dual Mesh Characteristic backtracked vertex position
  for i,j,k in Flags:
    if is_internal_vertex(i,j,k):
      dt = Dt[None]
      x,y,z = get_vert_loc(i,j,k)
      # x-direction
      a = 0.0
      if Vel_vert[i,j,k][0] <= 0:
        a = (Vel_vert[i,j,k][0] - Vel_vert[i-1,j,k][0])/dx
      else:
        a = -(Vel_vert[i,j,k][0] - Vel_vert[i+1,j,k][0])/dx

      if ti.abs(a) <= small:
        Vert_pos[i,j,k][0] = x - dt*Vel_vert[i,j,k][0]
      else:
        Vert_pos[i,j,k][0] = x - (1.0-ti.exp(-a*dt))*Vel_vert[i,j,k][0]/a

      # y-direction
      if Vel_vert[i,j,k][1] <= 0:
        a = (Vel_vert[i,j,k][1] - Vel_vert[i,j-1,k][1])/dy
      else:
        a = -(Vel_vert[i,j,k][1] - Vel_vert[i,j+1,k][1])/dy

      if ti.abs(a) <= small:
        Vert_pos[i,j,k][1] = y - dt*Vel_vert[i,j,k][0]
      else:
        Vert_pos[i,j,k][1] = y - (1.0-ti.exp(-a*dt))*Vel_vert[i,j,k][1]/a

      # z-direction
      if Vel_vert[i,j,k][2] <= 0:
        a = (Vel_vert[i,j,k][2] - Vel_vert[i,j,k-1][2])/dz
      else:
        a = -(Vel_vert[i,j,k][2] - Vel_vert[i,j,k+1][2])/dz

      if ti.abs(a) <= small:
        Vert_pos[i,j,k][2] = z - dt*Vel_vert[i,j,k][2]
      else:
        Vert_pos[i,j,k][2] = z - (1.0-ti.exp(-a*dt))*Vel_vert[i,j,k][2]/a


@ti.kernel
def compute_DC():
  # compute volume fraction fluxes using an isoadvector-like algorithm,
  # ie. the flux is the "time integral of the submerged face area"
  for i,j,k in Flags:
    dt = Dt[None]
    # flux the left face
    if is_internal_x_face(i,j,k) and is_active_x_face(i,j,k):
      # find the "upwind" interface cell
      iuw,juw,kuw = get_upwind_x(i,j,k)

      # intialize DCx to upwind volume fraction
      DCx[i,j,k] = C[iuw,juw,kuw]*U[i,j,k]*dy*dz
      if is_interface_cell(iuw,juw,kuw):
        # compute the level set at each vertex on this face at time t
        phi = [[[0.0,0.0],[0.0,0.0]],
               [[0.0,0.0],[0.0,0.0]]] # 3d array (y,z,t)
        x,y,z = get_vert_loc(i,j,k);
        phi[0][0][0] = get_phi_from_plic(x,y,z,iuw,juw,kuw)
        phi[1][0][0] = get_phi_from_plic(x,y+dy,z,iuw,juw,kuw)
        phi[0][1][0] = get_phi_from_plic(x,y,z+dz,iuw,juw,kuw)
        phi[1][1][0] = get_phi_from_plic(x,y+dy,z+dz,iuw,juw,kuw)

        # compute the level set at the lagrangian backtracked vertices
        phi[0][0][1] = get_phi_from_plic(Vert_pos[i,j,k][0],Vert_pos[i,j,k][1],Vert_pos[i,j,k][2],iuw,juw,kuw)
        phi[1][0][1] = get_phi_from_plic(Vert_pos[i,j+1,k][0],Vert_pos[i,j+1,k][1],Vert_pos[i,j+1,k][2],iuw,juw,kuw)
        phi[0][1][1] = get_phi_from_plic(Vert_pos[i,j,k+1][0],Vert_pos[i,j,k+1][1],Vert_pos[i,j,k+1][2],iuw,juw,kuw)
        phi[1][1][1] = get_phi_from_plic(Vert_pos[i,j+1,k+1][0],Vert_pos[i,j+1,k+1][1],Vert_pos[i,j+1,k+1][2],iuw,juw,kuw)

        #calculate the volume fraction of the space-time volume
        vf = calc_vol_frac(phi)
        DCx[i,j,k] = vf*U[i,j,k]*dy*dz

    # flux the bottom face
    if is_internal_y_face(i,j,k) and is_active_y_face(i,j,k):
      # find the "upwind" interface cell
      iuw,juw,kuw = get_upwind_y(i,j,k)

      # intialize DCy to upwind volume fraction
      DCy[i,j,k] = C[iuw,juw,kuw]*V[i,j,k]*dx*dz
      if is_interface_cell(iuw,juw,kuw):
        # compute the level set at each vertex on this face at time t
        phi = [[[0.0,0.0],[0.0,0.0]],
               [[0.0,0.0],[0.0,0.0]]] # 3d array (y,z,t)
        x,y,z = get_vert_loc(i,j,k);
        phi[0][0][0] = get_phi_from_plic(x,y,z,iuw,juw,kuw)
        phi[1][0][0] = get_phi_from_plic(x+dx,y,z,iuw,juw,kuw)
        phi[0][1][0] = get_phi_from_plic(x,y,z+dz,iuw,juw,kuw)
        phi[1][1][0] = get_phi_from_plic(x+dx,y,z+dz,iuw,juw,kuw)

        # compute the level set at the lagrangian backtracked vertices
        phi[0][0][1] = get_phi_from_plic(Vert_pos[i,j,k][0],Vert_pos[i,j,k][1],Vert_pos[i,j,k][2],iuw,juw,kuw)
        phi[1][0][1] = get_phi_from_plic(Vert_pos[i+1,j,k][0],Vert_pos[i+1,j,k][1],Vert_pos[i+1,j,k][2],iuw,juw,kuw)
        phi[0][1][1] = get_phi_from_plic(Vert_pos[i,j,k+1][0],Vert_pos[i,j,k+1][1],Vert_pos[i,j,k+1][2],iuw,juw,kuw)
        phi[1][1][1] = get_phi_from_plic(Vert_pos[i+1,j,k+1][0],Vert_pos[i+1,j,k+1][1],Vert_pos[i+1,j,k+1][2],iuw,juw,kuw)

        #calculate the volume fraction of the space-time volume
        vf = calc_vol_frac(phi)
        DCy[i,j,k] = vf*V[i,j,k]*dx*dz

    # flux the back face
    if is_internal_z_face(i,j,k) and is_active_z_face(i,j,k):
      # find the "upwind" interface cell
      iuw,juw,kuw = get_upwind_z(i,j,k)

      # intialize DCy to upwind volume fraction
      DCz[i,j,k] = C[iuw,juw,kuw]*W[i,j,k]*dx*dy
      if is_interface_cell(iuw,juw,kuw):
        # compute the level set at each vertex on this face at time t
        phi = [[[0.0,0.0],[0.0,0.0]],
               [[0.0,0.0],[0.0,0.0]]] # 3d array (y,z,t)
        x,y,z = get_vert_loc(i,j,k);
        phi[0][0][0] = get_phi_from_plic(x,y,z,iuw,juw,kuw)
        phi[1][0][0] = get_phi_from_plic(x+dx,y,z,iuw,juw,kuw)
        phi[0][1][0] = get_phi_from_plic(x,y+dy,z,iuw,juw,kuw)
        phi[1][1][0] = get_phi_from_plic(x+dx,y+dy,z,iuw,juw,kuw)

        # compute the level set at the lagrangian backtracked vertices
        phi[0][0][1] = get_phi_from_plic(Vert_pos[i,j,k][0],Vert_pos[i,j,k][1],Vert_pos[i,j,k][2],iuw,juw,kuw)
        phi[1][0][1] = get_phi_from_plic(Vert_pos[i+1,j,k][0],Vert_pos[i+1,j,k][1],Vert_pos[i+1,j,k][2],iuw,juw,kuw)
        phi[0][1][1] = get_phi_from_plic(Vert_pos[i,j+1,k][0],Vert_pos[i,j+1,k][1],Vert_pos[i,j+1,k][2],iuw,juw,kuw)
        phi[1][1][1] = get_phi_from_plic(Vert_pos[i+1,j+1,k][0],Vert_pos[i+1,j+1,k][1],Vert_pos[i+1,j+1,k][2],iuw,juw,kuw)

        #calculate the volume fraction of the space-time volume
        vf = calc_vol_frac(phi)
        DCz[i,j,k] = vf*W[i,j,k]*dx*dy



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
  """
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
  """
  return iup,jup,kup

@ti.func
def get_upwind_y(i,j,k):
  # get the "upwind" interface cell of this face
  # try the face neighbors first
  iup = i
  jup = j
  kup = k
  sgn = -1.0
  if V[i,j,k] > 0.0:
    jup = j-1
    sgn = 1.0
  """
  if not is_interface_cell(i,jup,k):
    # the face neighbor is not an interface cell
    # instead choose the edge or vertex neigbor with the highest upwind velocity
    for dk in ti.static(range(-1,2)):
      for di in ti.static(range(-1,2)):
        umax = 0.0
        if is_interface_cell(i+di,jup,k+dk) and sgn*V[i+di,jup,k+dk] > umax:
          iup = i+di
          kup = k+dk
          umax = sgn*V[i+di,jup,k+dk]
  """
  return iup,jup,kup

@ti.func
def get_upwind_z(i,j,k):
  # get the "upwind" interface cell of this face
  # try the face neighbors first
  iup = i
  jup = j
  kup = k
  sgn = -1.0
  if W[i,j,k] > 0.0:
    kup = k-1
    sgn = 1.0
  """
  if not is_interface_cell(i,jup,k):
    # the face neighbor is not an interface cell
    # instead choose the edge or vertex neigbor with the highest upwind velocity
    for dj in ti.static(range(-1,2)):
      for di in ti.static(range(-1,2)):
        umax = 0.0
        if is_interface_cell(i+di,j+dj,kup) and sgn*W[i+di,j+dj,kup] > umax:
          iup = i+di
          jup = j+dj
          umax = sgn*W[i+di,j+dj,kup]
  """
  return iup,jup,kup


@ti.func
def backtrack_dmc(x,y,z,i,j,k,dt):
  x_dmc = x-Vel_vert_dmc[i,j,k][0]*dt
  y_dmc = y-Vel_vert_dmc[i,j,k][1]*dt
  z_dmc = z-Vel_vert_dmc[i,j,k][2]*dt
  return x_dmc, y_dmc, z_dmc

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
  eps = 1.0e-20
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
        phi[0][j][k],phi[1][j][k] = phi[1][j][k],phi[0][j][k]
  if j0 == 1:
    for k in ti.static(range(2)):
      for i in ti.static(range(2)):
        phi[i][0][k],phi[i][1][k] = phi[i][1][k],phi[i][0][k]
  if k0 == 1:
    for j in ti.static(range(2)):
      for i in ti.static(range(2)):
        phi[i][j][0],phi[i][j][1] = phi[i][j][1],phi[i][j][0]

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
  order = [0,1,2]
  if l[0] < l[1]:
    order[0],order[1] = order[1],order[0]
    l[0],l[1] = l[1],l[0]
    Bx,By = By,Bx
    Bxz,Byz = Byz,Bxz
  if l[1] < l[2]:
    order[1],order[2] = order[2],order[1]
    l[1],l[2] = l[2],l[1]
    Bz,By = By,Bz
    Bxy,Bxz = Bxz,Bxy
  if l[0] < l[1]:
    order[0],order[1] = order[1],order[0]
    l[0],l[1] = l[1],l[0]
    Bx,By = By,Bx
    Bxz,Byz = Byz,Bxz

  #3 point 2d gaussian quadrature of z
  xq = [-np.sqrt(3.0/5.0), 0, np.sqrt(3.0/5.0)]; # quadrature points
  wq = [5.0/9.0, 8.0/9.0, 5.0/9.0];              # quadrature weights

  vf = 0.0
  Jx = l[0]/2.0 # jacobian
  z0 = 0.0
  for i in ti.static(range(3)):
    x = (xq[i]+1.0)*Jx
    # y integration bounds depends on x
    y1 = -((Bxz*z0 + Bx)*x + Bz*z0 + B) / ((Bxyz*z0 + Bxy)*x + Byz*z0 + By + eps)
    y1 = max(min(y1,1.0),0.0)
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
        if phi[0][j][k]*phi[1][j][k] < 0:
          li[j+2*k] = -phi[0][j][k]/(phi[1][j][k]-phi[0][j][k])
          ni+=1

    nj = 0
    lj = ti.Vector([1.0,1.0,1.0,1.0])
    for k in ti.static(range(2)):
      for i in ti.static(range(2)):
        if phi[i][0][k]*phi[i][1][k] < 0:
          lj[i+2*k] = -phi[i][0][k]/(phi[i][1][k]-phi[i][0][k])
          nj+=1
    nk = 0
    lk = ti.Vector([1.0,1.0,1.0,1.0])
    for j in ti.static(range(2)):
      for i in ti.static(range(2)):
        if phi[i][j][0]*phi[i][j][1] < 0:
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
      if not all_pos and not all_neg:
        vf += calc_vol_frac_b(phi)*(l[n]-l_old)
      elif all_pos:
        vf += l[n]-l_old

      l_old = l[n]

      #start next subcell at end of the last subcell
      for k in ti.static(range(2)):
        for j in ti.static(range(2)):
          phi[0][j][k] = phi[1][j][k]

  elif all_pos:
    vf = 1.0

  return vf


@ti.kernel
def update_C():
  for i,j,k in Flags:
    if is_active_cell(i,j,k):
      C[i,j,k] = C[i,j,k] + 1.0/vol*(DCx[i,j,k] - DCx[i+1,j,k] \
                          + DCy[i,j,k] - DCy[i,j+1,k] \
                          + DCz[i,j,k] - DCz[i,j,k+1])

@ti.kernel
def zero_DC_bounding():
  for i,j,k in Flags:
    DCx_b[i,j,k] = 0.0
    DCy_b[i,j,k] = 0.0
    DCz_b[i,j,k] = 0.0

@ti.kernel
def compute_DC_bounding():
  # redistribute C after advection such that it is not above one or below zero
  for i,j,k in Flags:
    if is_active_cell(i,j,k):
      dt = Dt[None]
      if C[i,j,k] > 1.0:
        # the extra C we need to redistribute to downwind cells
        extraC = (C[i,j,k]-1.0)*dx*dy*dz

        # sum of the downwind fluxes
        flux_x_0 = min(0.0,U[i,j,k]*dy*dz)
        flux_x_1 = max(0.0,U[i+1,j,k]*dy*dz)
        flux_y_0 = min(0.0,V[i,j,k]*dx*dz)
        flux_y_1 = max(0.0,V[i,j+1,k]*dx*dz)
        flux_z_0 = min(0.0,W[i,j,k]*dx*dy)
        flux_z_1 = max(0.0,W[i,j,k+1]*dx*dy)

        flux_sum = flux_x_0 + flux_x_1 + flux_y_0 + flux_y_1 + flux_z_0 + flux_z_1

        # compute redistribution deltaC
        if U[i,j,k] < 0.0:
          DCx_b[i,j,k] =  min(flux_x_0/flux_sum*extraC*vol, flux_x_0*dt-DCx[i,j,k])
        if U[i+1,j,k] > 0.0:
          DCx_b[i+1,j,k] =  min(flux_x_1/flux_sum*extraC*vol, flux_x_1*dt-DCx[i+1,j,k])
        if V[i,j,k] < 0.0:
          DCy_b[i,j,k] =  min(flux_y_0/flux_sum*extraC*vol, flux_y_0*dt-DCy[i,j,k])
        if V[i,j+1,k] > 0.0:
          DCy_b[i,j+1,k] =  min(flux_y_1/flux_sum*extraC*vol, flux_y_1*dt-DCx[i,j+1,k])
        if W[i,j,k] < 0.0:
          DCz_b[i,j,k] =  min(flux_z_0/flux_sum*extraC*vol, flux_z_0*dt-DCz[i,j,k])
        if W[i,j,k+1] > 0.0:
          DCz_b[i,j,k+1] =  min(flux_z_1/flux_sum*extraC*vol, flux_z_1*dt-DCz[i,j,k+1])

      if C[i,j,k] < 0.0:
        # the extra C we need to redistribute to downwind cells
        extraC = -C[i,j,k]*dx*dy*dz

        # sum of the downwind fluxes
        flux_x_0 = min(0.0,U[i,j,k]*dy*dz)
        flux_x_1 = max(0.0,U[i+1,j,k]*dy*dz)
        flux_y_0 = min(0.0,V[i,j,k]*dx*dz)
        flux_y_1 = max(0.0,V[i,j+1,k]*dx*dz)
        flux_z_0 = min(0.0,W[i,j,k]*dx*dy)
        flux_z_1 = max(0.0,W[i,j,k+1]*dx*dy)

        flux_sum = flux_x_0 + flux_x_1 + flux_y_0 + flux_y_1 + flux_z_0 + flux_z_1

        # compute redistribution deltaC
        if U[i,j,k] < 0.0:
          DCx_b[i,j,k] =  min(flux_x_0/flux_sum*extraC*vol, DCx[i,j,k])
        if U[i+1,j,k] > 0.0:
          DCx_b[i+1,j,k] =  min(flux_x_1/flux_sum*extraC*vol, DCx[i+1,j,k])
        if V[i,j,k] < 0.0:
          DCy_b[i,j,k] =  min(flux_y_0/flux_sum*extraC*vol, DCy[i,j,k])
        if V[i,j+1,k] > 0.0:
          DCy_b[i,j+1,k] =  min(flux_y_1/flux_sum*extraC*vol, DCx[i,j+1,k])
        if W[i,j,k] < 0.0:
          DCz_b[i,j,k] =  min(flux_z_0/flux_sum*extraC*vol, DCz[i,j,k])
        if W[i,j,k+1] > 0.0:
          DCz_b[i,j,k+1] =  min(flux_z_1/flux_sum*extraC*vol, DCz[i,j,k+1])

@ti.kernel
def update_C_bounding():
  for i,j,k in Flags:
    if is_active_cell(i,j,k):
      C[i,j,k] = C[i,j,k] + 1.0/vol*(DCx_b[i,j,k] - DCx_b[i+1,j,k] \
                          + DCy_b[i,j,k] - DCy_b[i,j+1,k] \
                          + DCz_b[i,j,k] - DCz_b[i,j,k+1])
