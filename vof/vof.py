from vof_data import *
from vof_initialize import *
from vof_narrowband import *
from vof_reconstruct import *
from vof_advect import *
from vof_visualize import *


def main():
	initialize()

	#flags.ptr.snode().parent.parent.clear_data_and_deactivate()
	#copyFlagsTempToFlags()
	#flagsTemp.ptr.snode().parent.parent.clear_data_and_deactivate()
  #copyFlagsToFlagsTemp()
	write_png()

if __name__ == '__main__':
  main()
