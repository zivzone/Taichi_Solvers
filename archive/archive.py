@ti.func
def swap(a,b):
  return b,a

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
  if l[0] < l[1]:
    l[0],l[1] = swap(l[0],l[1])
    Bx,By = swap(Bx,By)
    Bxz,Byz = swap(Bxz,Byz)
  if l[1] < l[2]:
    l[1],l[2] = swap(l[1],l[2])
    By,Bz = swap(By,Bz)
    Bxy,Bxz = swap(Bxy,Bxz)
  if l[0] < l[1]:
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
def calc_vol_frac_GQ(phi):
  # compute the volume fraction from level set at vertices
  # by splitting the cell into elementary cases then using gaussian quadrature
  # ref: 'A consistent analytical formulation for volume-estimation of geometries enclosed by
  # implicitly defined surfaces'
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
  def get_smooth_phi_and_weight_from_plic(x,y,z,i,j,k):
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

    w = phi*phi/r*r
    return phi, w

@ti.func
def calc_vol_frac(phi,grad_phi):
  # estimate volume fraction of cell using level set and its gradient
  phim = -ti.abs(phi)

  abs_grad_phi = [0.0,0.0,0.0]
  abs_grad_phi[0] = ti.abs(grad_phi[0])
  abs_grad_phi[1] = ti.abs(grad_phi[1])
  abs_grad_phi[2] = ti.abs(grad_phi[2])
  dxi   = ti.max(abs_grad_phi[0],ti.max(abs_grad_phi[1],abs_grad_phi[2]))
  dzeta = ti.min(abs_grad_phi[0],ti.min(abs_grad_phi[1],abs_grad_phi[2]))
  deta  = abs_grad_phi[0] + abs_grad_phi[1] + abs_grad_phi[2] - dxi - dzeta

  a = ti.max(phim + 0.5 * ( dxi + deta + dzeta), 0.0)
  b = ti.max(phim + 0.5 * ( dxi + deta - dzeta), 0.0)
  c = ti.max(phim + 0.5 * ( dxi - deta + dzeta), 0.0)
  d = ti.max(phim + 0.5 * (-dxi + deta + dzeta), 0.0)
  e = ti.max(phim + 0.5 * ( dxi - deta - dzeta), 0.0)
  vf = 0.0
  if dxi > 1.0e-8:
    if deta > 1.0e-8:
      if dzeta > 1.0e-8: #  3D
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

  if phi > 0.0:
    vf = 1.0 - vf

  return vf

@ti.kernel
def interp_velocity_to_vertex():
  # interpolate face center velocity components to cell vertices
  for i,j,k in Flags:
    if is_internal_vertex(i,j,k) or is_ghost_vertex(i,j,k):
      Vel_vert[i,j,k][0] = (U[i,j,k] + U[i-1,j,k])/2.0
      Vel_vert[i,j,k][1] = (V[i,j,k] + V[i,j-1,k])/2.0
      Vel_vert[i,j,k][2] = (W[i,j,k] + W[i,j,k-1])/2.0

@ti.kernel
def back_track_DMC():
  # compute the Dual Mesh Characteristic backtracked vertex position
  # ref 'Dual-Mesh Characteristics for Particle-Mesh Methods for the Simulation of Convection-Dominated Flows'
  # this is comparable in accuracy to rk4 but only takes one step
  for i,j,k in Flags:
    if is_internal_vertex(i,j,k):
      dt = Dt[None]
      x,y,z = get_vert_loc(i,j,k)
      # x-direction
      a = 0.0
      if Vel_vert[i,j,k][0] < 0.0:
        a = (Vel_vert[i,j,k][0] - Vel_vert[i-1,j,k][0])/dx
      else:
        a = -(Vel_vert[i,j,k][0] - Vel_vert[i+1,j,k][0])/dx

      if ti.abs(a) <= small:
        Vert_loc[i,j,k][0] = x - dt*Vel_vert[i,j,k][0]
      else:
        Vert_loc[i,j,k][0] = x - (1.0-ti.exp(-a*dt))*Vel_vert[i,j,k][0]/a

      # y-direction
      if Vel_vert[i,j,k][1] < 0.0:
        a = (Vel_vert[i,j,k][1] - Vel_vert[i,j-1,k][1])/dy
      else:
        a = -(Vel_vert[i,j,k][1] - Vel_vert[i,j+1,k][1])/dy

      if ti.abs(a) <= small:
        Vert_loc[i,j,k][1] = y - dt*Vel_vert[i,j,k][1]
      else:
        Vert_loc[i,j,k][1] = y - (1.0-ti.exp(-a*dt))*Vel_vert[i,j,k][1]/a

      # z-direction
      if Vel_vert[i,j,k][2] < 0.0:
        a = (Vel_vert[i,j,k][2] - Vel_vert[i,j,k-1][2])/dz
      else:
        a = -(Vel_vert[i,j,k][2] - Vel_vert[i,j,k+1][2])/dz

      if ti.abs(a) <= small:
        Vert_loc[i,j,k][2] = z - dt*Vel_vert[i,j,k][2]
      else:
        Vert_loc[i,j,k][2] = z - (1.0-ti.exp(-a*dt))*Vel_vert[i,j,k][2]/a
