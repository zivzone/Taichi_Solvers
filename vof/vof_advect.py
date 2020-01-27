from vof_data import *
from vof_util import *
from vof_reconstruct import *

@ti.kernel
def calc_Dt():
  ti.serialize()
  for i,j,k in Flags:
    if is_active_cell(i,j,k):
      dt = CFL*min(dx/(abs((U[i,j,k]+U[i+1,j,k])/2.0)+small), \
               min(dy/(abs((V[i,j,k]+V[i,j+1,k])/2.0)+small), \
                   dz/(abs((W[i,j,k]+W[i,j,k+1])/2.0)+small)))
      Dt[None] = min(Dt[None], dt)


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
      if Vel_vert[i,j,k][0] <= small:
        a = (Vel_vert[i,j,k][0] - Vel_vert[i-1,j,k][0])/dx
      else:
        a = -(Vel_vert[i,j,k][0] - Vel_vert[i+1,j,k][0])/dx

      if ti.abs(a) <= small:
        Vert_loc[i,j,k][0] = x - dt*Vel_vert[i,j,k][0]
      else:
        Vert_loc[i,j,k][0] = x - (1.0-ti.exp(-a*dt))*Vel_vert[i,j,k][0]/a

      # y-direction
      if Vel_vert[i,j,k][1] <= small:
        a = (Vel_vert[i,j,k][1] - Vel_vert[i,j-1,k][1])/dy
      else:
        a = -(Vel_vert[i,j,k][1] - Vel_vert[i,j+1,k][1])/dy

      if ti.abs(a) <= small:
        Vert_loc[i,j,k][1] = y - dt*Vel_vert[i,j,k][1]
      else:
        Vert_loc[i,j,k][1] = y - (1.0-ti.exp(-a*dt))*Vel_vert[i,j,k][1]/a

      # z-direction
      if Vel_vert[i,j,k][2] <= small:
        a = (Vel_vert[i,j,k][2] - Vel_vert[i,j,k-1][2])/dz
      else:
        a = -(Vel_vert[i,j,k][2] - Vel_vert[i,j,k+1][2])/dz

      if ti.abs(a) <= small:
        Vert_loc[i,j,k][2] = z - dt*Vel_vert[i,j,k][2]
      else:
        Vert_loc[i,j,k][2] = z - (1.0-ti.exp(-a*dt))*Vel_vert[i,j,k][2]/a


@ti.kernel
def compute_DC_isoadvector():
  # compute volume fraction fluxes using an isoadvector-like algorithm,
  # ie. the "time integral of the submerged face area".
  # To do this compute the levelset from the plic at the 4 vertices of the face at time t and t+dt,
  # then estimate the volume fraction of the 3-d space-time cell using analytical formulas
  # ref: 'A Computational Method for Sharp Interface Advection'
  for i,j,k in Flags:
    dt = Dt[None]
    # flux the left face
    if is_internal_x_face(i,j,k) and is_active_x_face(i,j,k):
      # get the "upwind" interface cell
      iuw = i
      juw = j
      kuw = k
      if U[i,j,k] > 0.0:
        iuw = i-1

      # intialize DCx to upwind volume fraction
      DCx[i,j,k] = C[iuw,juw,kuw]*U[i,j,k]*dy*dz*dt
      if is_interface_cell(iuw,juw,kuw):
        # compute the level set at each vertex on this face at time t
        phi = [[[0.0,0.0],[0.0,0.0]],
               [[0.0,0.0],[0.0,0.0]]] # 3d array (y,z,t)
        x,y,z = get_vert_loc(i,j,k);
        phi[0][0][0] = get_phi_from_plic(x,y,z,iuw,juw,kuw)
        phi[1][0][0] = get_phi_from_plic(x,y+dy,z,iuw,juw,kuw)
        phi[0][1][0] = get_phi_from_plic(x,y,z+dz,iuw,juw,kuw)
        phi[1][1][0] = get_phi_from_plic(x,y+dy,z+dz,iuw,juw,kuw)

        # compute the level set at the DMC backtracked vertices
        phi[0][0][1] = get_phi_from_plic(Vert_loc[i,j,k][0],Vert_loc[i,j,k][1],Vert_loc[i,j,k][2],iuw,juw,kuw)
        phi[1][0][1] = get_phi_from_plic(Vert_loc[i,j+1,k][0],Vert_loc[i,j+1,k][1],Vert_loc[i,j+1,k][2],iuw,juw,kuw)
        phi[0][1][1] = get_phi_from_plic(Vert_loc[i,j,k+1][0],Vert_loc[i,j,k+1][1],Vert_loc[i,j,k+1][2],iuw,juw,kuw)
        phi[1][1][1] = get_phi_from_plic(Vert_loc[i,j+1,k+1][0],Vert_loc[i,j+1,k+1][1],Vert_loc[i,j+1,k+1][2],iuw,juw,kuw)

        #calculate the volume fraction of the space-time volume
        alpha,m = calc_plic_from_phi(phi)
        DCx[i,j,k] = calc_C(alpha,m)*U[i,j,k]*dy*dz*dt

    # flux the bottom face
    if is_internal_y_face(i,j,k) and is_active_y_face(i,j,k):
      # get the "upwind" interface cell
      iuw = i
      juw = j
      kuw = k
      if V[i,j,k] > 0.0:
        juw = j-1

      # intialize DCy to upwind volume fraction
      DCy[i,j,k] = C[iuw,juw,kuw]*V[i,j,k]*dx*dz*dt
      if is_interface_cell(iuw,juw,kuw):
        # compute the level set at each vertex on this face at time t
        phi = [[[0.0,0.0],[0.0,0.0]],
               [[0.0,0.0],[0.0,0.0]]] # 3d array (x,z,t)
        x,y,z = get_vert_loc(i,j,k);
        phi[0][0][0] = get_phi_from_plic(x,y,z,iuw,juw,kuw)
        phi[1][0][0] = get_phi_from_plic(x+dx,y,z,iuw,juw,kuw)
        phi[0][1][0] = get_phi_from_plic(x,y,z+dz,iuw,juw,kuw)
        phi[1][1][0] = get_phi_from_plic(x+dx,y,z+dz,iuw,juw,kuw)

        # compute the level set at the lagrangian backtracked vertices
        phi[0][0][1] = get_phi_from_plic(Vert_loc[i,j,k][0],Vert_loc[i,j,k][1],Vert_loc[i,j,k][2],iuw,juw,kuw)
        phi[1][0][1] = get_phi_from_plic(Vert_loc[i+1,j,k][0],Vert_loc[i+1,j,k][1],Vert_loc[i+1,j,k][2],iuw,juw,kuw)
        phi[0][1][1] = get_phi_from_plic(Vert_loc[i,j,k+1][0],Vert_loc[i,j,k+1][1],Vert_loc[i,j,k+1][2],iuw,juw,kuw)
        phi[1][1][1] = get_phi_from_plic(Vert_loc[i+1,j,k+1][0],Vert_loc[i+1,j,k+1][1],Vert_loc[i+1,j,k+1][2],iuw,juw,kuw)

        #calculate the volume fraction of the space-time volume
        alpha,m = calc_plic_from_phi(phi)
        DCy[i,j,k] = calc_C(alpha,m)*V[i,j,k]*dx*dz*dt

    # flux the back face
    if is_internal_z_face(i,j,k) and is_active_z_face(i,j,k):
      # get the "upwind" interface cell
      iuw = i
      juw = j
      kuw = k
      if W[i,j,k] > 0.0:
        kuw = k-1

      # intialize DCy to upwind volume fraction
      DCz[i,j,k] = C[iuw,juw,kuw]*W[i,j,k]*dx*dy*dt
      if is_interface_cell(iuw,juw,kuw):
        # compute the level set at each vertex on this face at time t
        phi = [[[0.0,0.0],[0.0,0.0]],
               [[0.0,0.0],[0.0,0.0]]] # 3d array
        x,y,z = get_vert_loc(i,j,k);
        phi[0][0][0] = get_phi_from_plic(x,y,z,iuw,juw,kuw)
        phi[1][0][0] = get_phi_from_plic(x+dx,y,z,iuw,juw,kuw)
        phi[0][1][0] = get_phi_from_plic(x,y+dy,z,iuw,juw,kuw)
        phi[1][1][0] = get_phi_from_plic(x+dx,y+dy,z,iuw,juw,kuw)

        # compute the level set at the lagrangian backtracked vertices
        phi[0][0][1] = get_phi_from_plic(Vert_loc[i,j,k][0],Vert_loc[i,j,k][1],Vert_loc[i,j,k][2],iuw,juw,kuw)
        phi[1][0][1] = get_phi_from_plic(Vert_loc[i+1,j,k][0],Vert_loc[i+1,j,k][1],Vert_loc[i+1,j,k][2],iuw,juw,kuw)
        phi[0][1][1] = get_phi_from_plic(Vert_loc[i,j+1,k][0],Vert_loc[i,j+1,k][1],Vert_loc[i,j+1,k][2],iuw,juw,kuw)
        phi[1][1][1] = get_phi_from_plic(Vert_loc[i+1,j+1,k][0],Vert_loc[i+1,j+1,k][1],Vert_loc[i+1,j+1,k][2],iuw,juw,kuw)

        #calculate the volume fraction of the space-time volume
        alpha,m = calc_plic_from_phi(phi)
        DCz[i,j,k] = calc_C(alpha,m)*W[i,j,k]*dx*dy*dt


@ti.kernel
def update_C():
  for i,j,k in Flags:
    if is_active_cell(i,j,k):
      C[i,j,k] = C[i,j,k] + 1.0/vol*(DCx[i,j,k] - DCx[i+1,j,k] \
                          + DCy[i,j,k] - DCy[i,j+1,k] \
                          + DCz[i,j,k] - DCz[i,j,k+1])


@ti.kernel
def compute_DC_bounding():
  # redistribute C after advection in cases where the vof is very close to zero or one (above and below)
  for i,j,k in Flags:
    if is_active_cell(i,j,k):
      dt = Dt[None]
      # C after advection
      c_new = C[i,j,k] + 1.0/vol*(DCx[i,j,k] - DCx[i+1,j,k] \
                       + DCy[i,j,k] - DCy[i,j+1,k] \
                       + DCz[i,j,k] - DCz[i,j,k+1])

      # the extra C that needs to be redistributed to downwind cells
      c_extra = 0.0
      if c_new > c_one:
        c_extra = c_new-1.0
      elif c_new < c_zero:
        c_extra = c_new

      if c_new > c_one or c_new < c_zero:
        # note that for i faces flux is negative when downwind
        # and for i+1 faces flux is positive when downwind

        # sum of the downwind fluxes
        flux_x_0 = min(0.0,U[i,j,k]*dy*dz)
        flux_x_1 = max(0.0,U[i+1,j,k]*dy*dz)
        flux_y_0 = min(0.0,V[i,j,k]*dx*dz)
        flux_y_1 = max(0.0,V[i,j+1,k]*dx*dz)
        flux_z_0 = min(0.0,W[i,j,k]*dx*dy)
        flux_z_1 = max(0.0,W[i,j,k+1]*dx*dy)
        flux_sum = -flux_x_0 + flux_x_1 - flux_y_0 + flux_y_1 - flux_z_0 + flux_z_1

        if U[i,j,k] < 0.0:
          DCx[i,j,k] = min(max(DCx[i,j,k]+flux_x_0/flux_sum*c_extra*vol, flux_x_0*dt),0.0)
        if U[i+1,j,k] > 0.0:
          DCx[i+1,j,k] = max(min(DCx[i+1,j,k]+flux_x_1/flux_sum*c_extra*vol, flux_x_1*dt),0.0)
        if V[i,j,k] < 0.0:
          DCy[i,j,k] =  min(max(DCy[i,j,k]+flux_y_0/flux_sum*c_extra*vol, flux_y_0*dt),0.0)
        if V[i,j+1,k] > 0.0:
          DCy[i,j+1,k] = max(min(DCy[i,j+1,k]+flux_y_1/flux_sum*c_extra*vol, flux_y_1*dt),0.0)
        if W[i,j,k] < 0.0:
          DCz[i,j,k] =  min(max(DCz[i,j,k]+flux_z_0/flux_sum*c_extra*vol, flux_z_0*dt),0.0)
        if W[i,j+1,k] > 0.0:
          DCz[i,j,k+1] = max(min(DCz[i,j,k+1]+flux_z_1/flux_sum*c_extra*vol, flux_z_1*dt),0.0)


@ti.kernel
def cleanup_C():
  for i,j,k in Flags:
    if C[i,j,k] < c_zero:
      C[i,j,k] = 0.0
    if C[i,j,k] > c_one:
      C[i,j,k] = 1.0
