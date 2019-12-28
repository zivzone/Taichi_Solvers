from vof_data import *
from vof_initialize import *
from vof_band import *
from vof_reconstruct import *
from vof_advect import *
from vof_visualize import *

def main():
  initialize()
  write_C_png(0)
  for i in range(4):
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
    #zero_DC_bounding()
    #compute_DC_bounding()
    #update_C_bounding()

    write_band_png(i+1)
    write_C_png(i+1)
    write_M_png(i+1)
    write_Flag_png(i+1,flag_enum.CELL_INTERFACE, "interface_cell")
    write_Flag_png(i+1,flag_enum.CELL_ACTIVE, "active_cell")
    write_Flag_png(i+1,flag_enum.CELL_BUFFER, "buffer_cell")
    write_Flag_png(i+1,flag_enum.X_FACE_ACTIVE, "active_x_face")
    write_Flag_png(i+1,flag_enum.Y_FACE_ACTIVE, "active_y_face")
    write_Flag_png(i+1,flag_enum.Z_FACE_ACTIVE, "active_z_face")

    if i%10 ==0:
      print(i)



  ti.profiler_print()
if __name__ == '__main__':
  main()
