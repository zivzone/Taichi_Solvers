from vof_data import *

@ti.kernel
def copy_to_temp():
	for i,j,k in Flags:
		Flags_temp[i,j,k] = Flags[i,j,k]
		C_temp[i,j,k] = C[i,j,k]

@ti.kernel
def copy_from_temp():
	for i,j,k in Flags_temp:
		Flags[i,j,k] = Flags_temp[i,j,k]
		C[i,j,k] = C_temp[i,j,k]

def clear_and_deactivate_band():
	Flags.ptr.snode().parent.parent.clear_data_and_deactivate()

def clear_and_deactivate_band_temp():
	Flags_temp.ptr.snode().parent.parent.clear_data_and_deactivate()

@ti.kernel
def grow_interface_band():
	for i,j,k in Flags_temp:
		if (i!=0 and j!=0 and k!=0):
			if Flags_temp[i,j,k]&cellFlags.CELL_ACTIVE==cellFlags.CELL_ACTIVE:
				# check if this is and interface cell
				if (C_temp[i,j,k] >= Czero and C_temp[i,j,k] <= Cone):
					Flags[i,j,k] = cellFlags.CELL_INTERFACE
					C[i,j,k] = C_temp[i,j,k]
				else:
					# treat cases where interface sits on cell face
					if (C_temp[i,j,k] > Cone):
						if (C_temp[i-1,j,k] < Czero or C_temp[i,j-1,k] < Czero or C_temp[i,j,k-1] < Czero):
							Flags[i,j,k] = cellFlags.CELL_INTERFACE
							C[i,j,k] = C_temp[i,j,k]
					elif (C_temp[i,j,k] < Czero):
						if (C_temp[i-1,j,k] > Cone or C_temp[i,j-1,k] > Cone or C_temp[i,j,k-1] > Cone):
							Flags[i,j,k] = cellFlags.CELL_INTERFACE
							C[i,j,k] = C_temp[i,j,k]

@ti.kernel
def grow_active_band():
	for i,j,k in Flags:
		if (i!=0 and j!=0 and k!=0 and i!=n_x and j!=n_y and k!=n_z):
			if Flags[i,j,k]&cellFlags.CELL_INTERFACE==cellFlags.CELL_INTERFACE:
				# flag this cell and neighbors as active
					for di in ti.static(range(-1,2)):
						for dj in ti.static(range(-1,2)):
							for dk in ti.static(range(-1,2)):
								Flags[i+di,j+dj,k+dk] = Flags[i+di,j+dj,k+dk]|cellFlags.CELL_ACTIVE
								C[i+di,j+dj,k+dk] = C_temp[i+di,j+dj,k+dk]

@ti.kernel
def grow_ghost_band():
	for i,j,k in Flags:
		if (i!=0 and j!=0 and k!=0 and i!=n_x and j!=n_y and k!=n_z):
			if Flags[i,j,k]&cellFlags.CELL_ACTIVE==cellFlags.CELL_ACTIVE:
				# flag neighbors as ghost if they arnt active
					for di in ti.static(range(-1,2)):
						for dj in ti.static(range(-1,2)):
							for dk in ti.static(range(-1,2)):
								if Flags[i+di,j+dj,k+dk]&cellFlags.CELL_ACTIVE!=cellFlags.CELL_ACTIVE:
									Flags[i+di,j+dj,k+dk] = cellFlags.CELL_GHOST
									C[i+di,j+dj,k+dk] = C[i,j,k]
