from vof_data import *
from vof_initialize import *
from vof_narrowband import *
from vof_reconstruct import *
from vof_advect import *
from vof_visualize import *

def main():

	initialize()

	copy_to_temp()
	clear_data_and_deactivate()
	grow_interface_band()
	grow_active_band()
	grow_ghost_band()

	reconstruct_plic()

	#write_Flags_png(0)
	#write_C_png(0)
	#write_M_png(0)

	ti.profiler_print()
if __name__ == '__main__':
  main()
