from vof_data import *
from vof_initialize import *
from vof_band import *
from vof_reconstruct import *
from vof_advect import *
from vof_boundary import *
from vof_visualize import *

def main():
  initialize()
  write_C_png(0)
  for i in range(200):
    # update the narrow band
    copy_to_temp()
    clear_data()
    grow_band()
    clear_data_temp()

    # reconstruct the interface
    reconstruct_plic()

    # advect the volume fraction
    set_face_velocity()
    interp_face_velocity_to_vertex()
    Dt[None] = .5*dx
    back_track_DMC()
    compute_DC()
    update_C()

    # bound volume fraction to physical values
    #for j in range(5):
    #  zero_DC_bounding()
    #  compute_DC_bounding()
    #  update_C_bounding()
    clamp_C()

    #apply boundary conditions
    apply_Neumann_BC()

    if i%5 ==0:
      print(i)
      write_band_png(i+1)
      write_C_png(i+1)
      write_M_png(i+1)

  plot_interfaces()

  ti.profiler_print()
if __name__ == '__main__':
  main()
