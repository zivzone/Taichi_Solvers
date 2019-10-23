from vof_data import *

@ti.kernel
def compute_dC():
  for i,j,k in Flags:
    if Flags[i,j,k]&flag_enum.CELL_ACTIVE==flag_enum.CELL_ACTIVE:
      if is_internal_cell(i,j,k):
        # flux the left face
        #**********************************************************************
        # find the "upwind" interface cell
        iup,jup,kup = get_upwind_x(i,j,k)
        dCx[i,j,k] = C[iup,jup,kup]
        if is_interface(iup,jup,kup):
          phi = [[0.0,0.0],[0.0,0.0],[0.0,0.0]]


@ti.kernel
def interp_face_velocity_to_vertex():
  # interpolate face center volocity components to cell vertices
  for i,j,k in Flags:
    if is_internal_cell(i,j,k):
      Vel_vert[i,j,k][0] = (U[i,j,k] + U[i-1,j,k])/2.0
      Vel_vert[i,j,k][1] = (V[i,j,k] + V[i,j-1,k])/2.0
      Vel_vert[i,j,k][2] = (W[i,j,k] + W[i,j,k-1])/2.0


@ti.kernel
def set_face_velocity():
  # set left/bottom/back face velocities from preset field
  for i,j,k in Flags:
    x,y,z = get_cell_loc(i,j,k)

    u,v,w = get_vel(x-dx/2.0,y,z) # at face loc
    U[i,j,k] = u

    u,v,w = get_vel(x,y-dy/2.0,z) # at face loc
    V[i,j,k] = v

    u,v,w = get_vel(x,y,z-dz/2.0) # at face loc
    W[i,j,k] = w

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

## get_velocity functions ##
@ti.func
def get_vel_solid_body_rotation(x,y,z):
  u = 0.5 - y/w_y
  v = -0.5 + x/w_x
  w = 0.0
  return u,v,w

@ti.func
def get_vel_vortex_in_a_box(x,y,z):
  xpi = np.pi*x/w_x
  ypi = np.pi*y/w_x
  u = -2.0*ti.sin(xpi)*ti.sin(xpi)*ti.sin(ypi)*ti.cos(ypi)
  v =  2.0*ti.sin(ypi)*ti.sin(ypi)*ti.sin(xpi)*ti.cos(xpi)
  w =  0.0
  return u,v,w

@ti.func
def get_vel_transport(x,y,z):
  u =  1.0
  v =  0.0
  w =  0.0
  return u,v,w

get_vel = get_vel_transport
