from vof_data import *
from vof_initialize import *
from vof_band import *
from vof_reconstruct import *
from vof_advect import *
from vof_visualize import *

def main():

  initialize()

  for i in range(1000):

    # update the narrow band
    copy_to_temp()
    clear_data_and_deactivate()
    grow_interface_band()
    grow_active_band()
    grow_ghost_band()
    clear_data_and_deactivate_temp()

    # reconstruct the interface
    reconstruct_plic()

    # advect the volume fraction
    #set_face_velocity()
    #interp_face_velocity_to_vertex()
    if i%10 ==0:
      print(i)
  # output pngs
  write_Flags_png(0)
  write_C_png(0)
  write_M_png(0)

  ti.profiler_print()
if __name__ == '__main__':
  main()
