from vof_data import *
from vof_common import *

@ti.kernel
def compute_DC(dt):
  # compute volume fraction fluxes using an isoadvector-like algorithm,
  # the flux is the "time integral of the submerged face area"
  for i,j,k in Flags:
    # flux the left face
    if is_internal_x_face(i,j,k) and is_active_x_face(i,j,k):
      # find the "upwind" interface cell
      iup,jup,kup = get_upwind_x(i,j,k)

      # intialize DCx to upwind volume fraction
      DCx[i,j,k] = C[iup,jup,kup]
      if is_interface_cell(iuw,juw,kuw):
        # compute the level set at each vertex on this face at time t
        phi = [[0.0,0.0],[0.0,0.0],[0.0,0.0]] # (x,y,t)
        x,y,z = get_vert_loc(i,j,k);
        phi[0,0,0] = get_phi_from_plic(x,y,z,iuw,juw,kuw)
        phi[1,0,0] = get_phi_from_plic(x,y+dy,z,iuw,juw,kuw)
        phi[0,1,0] = get_phi_from_plic(x,y,z+dz,iuw,juw,kuw)
        phi[1,1,0] = get_phi_from_plic(x,y+dy,z+dz,iuw,juw,kuw)

        # lagrangian backtrack the vertices using DMC to compute the level set at time t+dt
        x_dmc,y_dmc,z_dmc = backtrack_DMC(x,y,z,i,j,k)
        phi[0,0,1] = get_phi_from_plic(x_dmc,y_dmc,z_dmc,iuw,juw,kuw)
        x_dmc,y_dmc,z_dmc = backtrack_DMC(x,y+dy,z,i,j+1,k)
        phi[1,0,1] = get_phi_from_plic(x_dmc,y_dmc,z_dmc,iuw,juw,kuw)
        x_dmc,y_dmc,z_dmc = backtrack_DMC(x,y,z+dz,i,j,k+1)
        phi[0,1,1] = get_phi_from_plic(x_dmc,y_dmc,z_dmc,iuw,juw,kuw)
        x_dmc,y_dmc,z_dmc = backtrack_DMC(x,y+dz,z+dz,i,j+1,k+1)
        phi[1,1,1] = get_phi_from_plic(x_dmc,y_dmc,z_dmc,iuw,juw,kuw)




@ti.kernel
def interp_face_velocity_to_vertex():
  # interpolate face center volocity components to cell vertices
  for i,j,k in Flags:
    if is_internal_vertex(i,j,k):
      Vel_vert[i,j,k][0] = (U[i,j,k] + U[i-1,j,k])/2.0
      Vel_vert[i,j,k][1] = (V[i,j,k] + V[i,j-1,k])/2.0
      Vel_vert[i,j,k][2] = (W[i,j,k] + W[i,j,k-1])/2.0

@ti.func
def backtrack_DMC(x,y,z,i,j,k):
  # x-direction
  if Vel_vert[i,j,k][0] <=0:
    a = (Vel_vert[i,j,k][0] - Vel_vert[i-1,j,k][0])/dx
  else:
    a = -(Vel_vert[i,j,k][0] - Vel_vert[i+1,j,k][0])/dx

  if a == 0:
    x_dmc = x - Vel_vert[i,j,k][0]*dt
  else:
    x_dmc = x - (1.0-exp(-a*dt))*Vel_vert[i,j,k][0]/a

  # y-direction
  if Vel_vert[i,j,k][1] <=0:
    a = (Vel_vert[i,j,k][1] - Vel_vert[i-1,j,k][1])/dy
  else:
    a = -(Vel_vert[i,j,k][1] - Vel_vert[i+1,j,k][1])/dy

  if a == 0:
    y_dmc = y - Vel_vert[i,j,k][1]*dt
  else:
    y_dmc = y - (1.0-exp(-a*dt))*Vel_vert[i,j,k][1]/a

  # z-direction
  if Vel_vert[i,j,k][2] <=0:
    a = (Vel_vert[i,j,k][2] - Vel_vert[i-1,j,k][2])/dz
  else:
    a = -(Vel_vert[i,j,k][2] - Vel_vert[i+1,j,k][2])/dz

  if a == 0:
    z_dmc = z - Vel_vert[i,j,k][2]*dt
  else:
    z_dmc = z - (1.0-exp(-a*dt))*Vel_vert[i,j,k][2]/a

  return x,y,z


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
