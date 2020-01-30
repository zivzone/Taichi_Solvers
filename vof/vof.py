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
    clear()
    grow_band()
    clear_temp()

    # set the face velocities and compute timestep
    set_face_velocity()

    Dt[None] = big
    calc_Dt()
    if t+Dt[None] > t_final:
      Dt[None] = t_final-t

    # reconstruct the interface
    reconstruct_plic()

    Tot_vol[None] = 0.0
    check_vof()
    if i == 0:
      init_vol = Tot_vol[None]

    vol_err = init_vol-Tot_vol[None]

    # advect the volume fraction
    interp_velocity_to_vertex()
    back_track_DMC()
    compute_DC_isoadvector()
    for j in range(10):
      compute_DC_bounding()
    update_C()
    cleanup_C()

    #apply boundary conditions
    apply_Neumann_BC()


    if i%plot_interval==0:
      #print(Dt[None])
      print(i)
      print(vol_err)
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
