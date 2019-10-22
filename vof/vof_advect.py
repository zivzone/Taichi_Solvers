from vof_data import *

@ti.kernel
def compute_dC():
	for i,j,k in Flags:
		if Flags[i,j,k]&cell_flags.CELL_ACTIVE==cell_flags.CELL_ACTIVE:
			if is_internal(i,j,k):
				a = 1

@ti.kernel
def interp_face_velocity_to_vertex():
	# interpolate face center volocity components to cell vertices
	for i,j,k in Flags:
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
