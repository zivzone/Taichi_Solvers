from vof_data import *
from vof_initialize import *
from vof_narrowband import *
from vof_reconstruct import *
from vof_advect import *
from vof_visualize import *

def main():
	initialize()

	copy_to_temp()
	clear_and_deactivate_band()
	grow_interface_band()
	grow_active_band()
	grow_ghost_band()

	reconstruct_plic()

	write_Flags_png('Flags0.png')
	write_C_png('C0.png')
	write_M_png('M0.png')

if __name__ == '__main__':
  main()
