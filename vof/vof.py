from vof_data import *
from vof_initialize import *
from vof_narrowband import *
from vof_reconstruct import *
from vof_advect import *
from vof_visualize import *


def main():
	initialize()
	write_png('slice0.png')

	copy_to_temp()
	clear_and_deactivate_band()
	grow_interface_band()
	grow_active_band()
	grow_ghost_band()

	write_png('slice1.png')

if __name__ == '__main__':
  main()
