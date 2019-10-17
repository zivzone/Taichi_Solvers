import taichi as ti
from vof_data import *

@ti.kernel
def copy_temp_to_data():
	for i,j,k in Flags:
		Flags_temp[i,j,k] = Flags[i,j,k]
		C_temp[i,j,k] = C[i,j,k]

@ti.kernel
def copy_data_to_temp():
	for i,j,k in Flags_temp:
		Flags[i,j,k] = Flags_temp[i,j,k]
		C[i,j,k] = C_temp[i,j,k]
