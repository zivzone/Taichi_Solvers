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

@ti.kernel
def grow_interface_band():
	for i,j,k in Flags_temp:
		if is_internal(i,j,k):
			if Flags_temp[i,j,k]&cell_flags.CELL_ACTIVE==cell_flags.CELL_ACTIVE:
				# check if this is and interface cell
				if (C_temp[i,j,k] >= Czero and C_temp[i,j,k] <= Cone):
					Flags[i,j,k] = cell_flags.CELL_INTERFACE
					C[i,j,k] = C_temp[i,j,k]
				else:
					# treat cases where interface sits on cell face
					if (C_temp[i,j,k] > Cone):
						if (C_temp[i-1,j,k] < Czero or C_temp[i,j-1,k] < Czero or C_temp[i,j,k-1] < Czero):
							Flags[i,j,k] = cell_flags.CELL_INTERFACE
							C[i,j,k] = C_temp[i,j,k]
					elif (C_temp[i,j,k] < Czero):
						if (C_temp[i-1,j,k] > Cone or C_temp[i,j-1,k] > Cone or C_temp[i,j,k-1] > Cone):
							Flags[i,j,k] = cell_flags.CELL_INTERFACE
							C[i,j,k] = C_temp[i,j,k]

@ti.kernel
def grow_active_band():
	for i,j,k in Flags:
		if is_internal(i,j,k):
			if Flags[i,j,k]&cell_flags.CELL_INTERFACE==cell_flags.CELL_INTERFACE:
				# flag this cell and neighbors as active
					for di in ti.static(range(-1,2)):
						for dj in ti.static(range(-1,2)):
							for dk in ti.static(range(-1,2)):
								Flags[i+di,j+dj,k+dk] = Flags[i+di,j+dj,k+dk]|cell_flags.CELL_ACTIVE
								C[i+di,j+dj,k+dk] = C_temp[i+di,j+dj,k+dk]

@ti.kernel
def grow_ghost_band():
	for i,j,k in Flags:
		if is_internal(i,j,k):
			if Flags[i,j,k]&cell_flags.CELL_ACTIVE==cell_flags.CELL_ACTIVE:
				# flag neighbors as ghost if they arnt active
					for di in ti.static(range(-1,2)):
						for dj in ti.static(range(-1,2)):
							for dk in ti.static(range(-1,2)):
								if Flags[i+di,j+dj,k+dk]&cell_flags.CELL_ACTIVE!=cell_flags.CELL_ACTIVE:
									Flags[i+di,j+dj,k+dk] = cell_flags.CELL_GHOST
									C[i+di,j+dj,k+dk] = C[i,j,k]