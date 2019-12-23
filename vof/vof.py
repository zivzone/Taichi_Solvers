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
    clear_data()
    grow_interface_band()
    grow_active_band()
    grow_buffer_band()
    clear_data_temp()

    # reconstruct the interface
    reconstruct_plic()

    # advect the volume fraction
    #set_face_velocity()
    #interp_face_velocity_to_vertex()
    #compute_dmc_velocity(.01)
    #compute_DC(.01)

    if i%10 ==0:
      print(i)
  # output pngs

  write_Flags_png(0)
  write_C_png(0)
  write_M_png(0)

  ti.profiler_print()
if __name__ == '__main__':
  main()
