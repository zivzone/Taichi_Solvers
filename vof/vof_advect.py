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
        x = x-U[i,j,k]*dt
        y = y-(V[i,j,k]+V[i-1,j,k]+V[i,j+1,k]+V[i-1,j+1,k])/4.0*dt
        z = z-(W[i,j,k]+W[i-1,j,k]+W[i,j,k+1]+W[i-1,j,k+1])/4.0*dt
        phi[0][0][1] = get_phi_from_plic(x,y,z,iuw,juw,kuw)
        phi[1][0][1] = get_phi_from_plic(x,y+dy,z,iuw,juw,kuw)
        phi[0][1][1] = get_phi_from_plic(x,y,z+dz,iuw,juw,kuw)
        phi[1][1][1] = get_phi_from_plic(x,y+dy,z+dz,iuw,juw,kuw)

        #calculate the delta C from the volume fraction of the space-time volume
        alpha,m = calc_plic_from_phi(phi)
        c = calc_C(alpha,m)
        DCx[i,j,k] = c*U[i,j,k]*dy*dz*dt

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
        x = x-(U[i,j,k]+U[i,j-1,k]+U[i+1,j,k]+U[i+1,j-1,k])/4.0*dt
        y = y-V[i,j,k]*dt
        z = z-(W[i,j,k]+W[i,j-1,k]+W[i,j,k+1]+W[i,j-1,k+1])/4.0*dt
        phi[0][0][1] = get_phi_from_plic(x,y,z,iuw,juw,kuw)
        phi[1][0][1] = get_phi_from_plic(x+dx,y,z,iuw,juw,kuw)
        phi[0][1][1] = get_phi_from_plic(x,y,z+dz,iuw,juw,kuw)
        phi[1][1][1] = get_phi_from_plic(x+dx,y,z+dz,iuw,juw,kuw)

        #calculate the delta C from the volume fraction of the space-time volume
        alpha,m = calc_plic_from_phi(phi)
        c = calc_C(alpha,m)
        DCy[i,j,k] = c*V[i,j,k]*dx*dz*dt

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
        x = x-(U[i,j,k]+U[i,j,k-1]+U[i+1,j,k]+U[i+1,j,k-1])/4.0*dt
        y = y-(V[i,j,k]+V[i,j,k-1]+V[i,j+1,k]+V[i,j+1,k-1])/4.0*dt
        z = z-W[i,j,k]*dt
        phi[0][0][1] = get_phi_from_plic(x,y,z,iuw,juw,kuw)
        phi[1][0][1] = get_phi_from_plic(x+dx,y,z,iuw,juw,kuw)
        phi[0][1][1] = get_phi_from_plic(x,y+dy,z,iuw,juw,kuw)
        phi[1][1][1] = get_phi_from_plic(x+dx,y+dy,z,iuw,juw,kuw)

        #calculate the delta C from the volume fraction of the space-time volume
        alpha,m = calc_plic_from_phi(phi)
        c = calc_C(alpha,m)
        DCz[i,j,k] = c*W[i,j,k]*dx*dy*dt


@ti.kernel
def update_C():
  for i,j,k in Flags:
    if is_active_cell(i,j,k):
      C[i,j,k] = C[i,j,k] + 1.0/vol*(DCx[i,j,k] - DCx[i+1,j,k] \
                          + DCy[i,j,k] - DCy[i,j+1,k] \
                          + DCz[i,j,k] - DCz[i,j,k+1])


@ti.kernel
def compute_DC_bounding():
  # modify fluxes to redistribute C after advection in cases where the vof is very close to zero or one (above and below)
  for i,j,k in Flags:
    if is_active_cell(i,j,k):
      dt = Dt[None]
      # C after advection
      c_new = C[i,j,k] + 1.0/vol*(DCx[i,j,k] - DCx[i+1,j,k] \
                       + DCy[i,j,k] - DCy[i,j+1,k] \
                       + DCz[i,j,k] - DCz[i,j,k+1])

      # the extra C that needs to be distributed to downwind cells
      c_extra = 0.0
      if c_new > c_one:
        c_extra = (c_new-1.0)*vol
      elif c_new < c_zero:
        c_extra = c_new*vol

      if c_new > c_one or c_new < c_zero:
        # face weights are the amount of "available" dC flux
        # "available" dC flux is the difference of the current dC and max/min dC
        # max is vel*face_area*dt, min is 0
        wt_x_0 = 0.0
        if U[i,j,k] < 0.0 and is_active_cell(i-1,j,k):
          if c_extra > 0.0:
            wt_x_0 = U[i,j,k]*dy*dz*dt-DCx[i,j,k]
          else:
            wt_x_0 = DCx[i,j,k]

        wt_x_1 = 0.0
        if U[i+1,j,k] > 0.0 and is_active_cell(i+1,j,k):
          if c_extra > 0.0:
            wt_x_1 = U[i+1,j,k]*dy*dz*dt-DCx[i+1,j,k]
          else:
            wt_x_1 = DCx[i+1,j,k]

        wt_y_0 = 0.0
        if V[i,j,k] < 0.0 and is_active_cell(i,j-1,k):
          if c_extra > 0.0:
            wt_y_0 = V[i,j,k]*dx*dz*dt-DCy[i,j,k]
          else:
            wt_y_0 = DCy[i,j,k]

        wt_y_1 = 0.0
        if V[i,j+1,k] > 0.0 and is_active_cell(i,j+1,k):
          if c_extra > 0.0:
            wt_y_1 = V[i,j+1,k]*dx*dz*dt-DCy[i,j+1,k]
          else:
            wt_y_1 = DCy[i,j+1,k]

        wt_z_0 = 0.0
        if W[i,j,k] < 0.0 and is_active_cell(i,j,k-1):
          if c_extra > 0.0:
            wt_z_0 = W[i,j,k]*dx*dy*dt-DCz[i,j,k]
          else:
            wt_z_0 = DCz[i,j,k]

        wt_z_1 = 0.0
        if W[i,j,k+1] > 0.0 and is_active_cell(i,j,k+1):
          if c_extra > 0.0:
            wt_z_1 = W[i,j,k+1]*dx*dy*dt-DCz[i,j,k+1]
          else:
            wt_z_1 = DCx[i,j,k+1]

        wt_sum = -wt_x_0 + wt_x_1 - wt_y_0 + wt_y_1 - wt_z_0 + wt_z_1

        # sum of the downwind fluxes
        # note that downwind is negative for i faces and positive for i+1 face
        #flux_x_0 = min(0.0,U[i,j,k]*dy*dz)
        #flux_x_1 = max(0.0,U[i+1,j,k]*dy*dz)
        #flux_y_0 = min(0.0,V[i,j,k]*dx*dz)
        #flux_y_1 = max(0.0,V[i,j+1,k]*dx*dz)
        #flux_z_0 = min(0.0,W[i,j,k]*dx*dy)
        #flux_z_1 = max(0.0,W[i,j,k+1]*dx*dy)
        #flux_sum = -flux_x_0 + flux_x_1 - flux_y_0 + flux_y_1 - flux_z_0 + flux_z_1

        # modify deltaC's to redistribute the extra c to active downwind cells weighted by the face flux.
        # limit the dC bewtween 0 and 100% of the flux volume
        if wt_x_0 != 0.0:
          DCx[i,j,k] = min(max(DCx[i,j,k]+wt_x_0/wt_sum*c_extra, U[i,j,k]*dy*dz*dt),0.0)

        if wt_x_1 != 0.0:
          DCx[i+1,j,k] = max(min(DCx[i+1,j,k]+wt_x_1/wt_sum*c_extra, U[i+1,j,k]*dy*dz*dt),0.0)

        if wt_y_0 != 0.0:
          DCy[i,j,k] =  min(max(DCy[i,j,k]+wt_y_0/wt_sum*c_extra, V[i,j,k]*dx*dz*dt),0.0)

        if wt_y_1 != 0.0:
          DCy[i,j+1,k] = max(min(DCy[i,j+1,k]+wt_y_1/wt_sum*c_extra, V[i,j+1,k]*dx*dz*dt),0.0)

        if wt_z_0 != 0.0:
          DCz[i,j,k] =  min(max(DCz[i,j,k]+wt_z_0/wt_sum*c_extra, W[i,j,k]*dx*dy*dt),0.0)

        if wt_z_1 != 0.0:
          DCz[i,j,k+1] = max(min(DCz[i,j,k+1]+wt_z_1/wt_sum*c_extra, W[i,j,k+1]*dx*dy*dt),0.0)



@ti.kernel
def cleanup_C():
  for i,j,k in Flags:
    if C[i,j,k] < c_zero:
      C[i,j,k] = 0.0
    if C[i,j,k] > c_one:
      C[i,j,k] = 1.0
