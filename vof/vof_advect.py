from vof_data import *
from vof_common import *
"""
@ti.kernel
def compute_DC():
  # compute volume fraction fluxes using isoadvector-like algorithm
  for i,j,k in Flags:
    # flux the left face
    if is_internal_face(i,j,k) and is_active_x_face(i,j,k):
      # find the "upwind" interface cell
      iup,jup,kup = get_upwind_x(i,j,k)

      # intialize DCx to upwind volume fraction
      DCx[i,j,k] = C[iup,jup,kup]
      if is_interface(iup,jup,kup):
        # compute the "time integral of submerged face area"
        phi = [[0.0,0.0],[0.0,0.0],[0.0,0.0]] # (x,y,t)


@ti.kernel
def interp_face_velocity_to_vertex():
  # interpolate face center volocity components to cell vertices
  for i,j,k in Flags:
    if is_internal_cell(i,j,k):
      Vel_vert[i,j,k][0] = (U[i,j,k] + U[i-1,j,k])/2.0
      Vel_vert[i,j,k][1] = (V[i,j,k] + V[i,j-1,k])/2.0
      Vel_vert[i,j,k][2] = (W[i,j,k] + W[i,j,k-1])/2.0


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
    # the face neghbor is not an interface cell
    # instead choose the edge or vertex neigbor with the highest upwind velocity
    for dk in range(-1,2):
      for dj in range(-1,2):
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

  # phi is distance form plic plane
  phi = (M[i,j,k][0]*x + M[i,j,k][1]*y + M[i,j,k][2]*z - Alpha[i,j,k])/np.sqrt(3.0)
  return phi
"""
