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
  t = 0
  i = 0

  while t < t_final:
    # update the narrow band
    copy_to_temp()
    clear_data()
    grow_band()
    clear_data_temp()

    # set the face velocities and compute timestep
    set_face_velocity()
    Dt[None] = big
    calc_Dt()
    if t+Dt[None] > t_final:
      Dt[None] = t_final-t

    # reconstruct the interface
    reconstruct_plic()
    check_vof()

    # advect the volume fraction
    interp_velocity_to_vertex()
    back_track_DMC()
    compute_DC_isoadvector()
    update_C()

    # bound volume fraction to physical values
    #for j in range(5):
    #  zero_DC_bounding()
    #  compute_DC_bounding()
    #  update_C_bounding()
    cleanup_C()

    #apply boundary conditions
    apply_Neumann_BC()

    print(i)
    if i%plot_interval==0:
      print(Dt[None])
      write_band_png(i+1)
      write_C_png(i+1)
      #write_M_png(i+1)

    t += Dt[None]
    i += 1

  print(t)
  write_band_png(i+1)
  write_C_png(i+1)
  #write_M_png(i+1)
  plot_interfaces()
  plt.show()




if __name__ == '__main__':
  main()
