import numpy as np
import matplotlib.pyplot as plt
import cv2
from vof_data import *
from vof_util import *


@ti.func
def subtract_ext(i,j):
  return i-nx_ext, j-ny_ext


@ti.kernel
def band_to_img(img: ti.ext_arr()):
  for i,j,k in Flags:
    if is_internal_cell(i,j,k):
      if k == nz_ext:
        ii,jj = subtract_ext(i,j)
        if is_interface_cell(i,j,k):
          img[ii,jj] = 1.0
        elif is_active_cell(i,j,k):
          img[ii,jj] = 2.0/3.0
        elif is_buffer_cell(i,j,k):
          img[ii,jj] = 1.0/3.0
        else:
          img[ii,jj] = 1.0/9.0

@ti.kernel
def Flag_to_img(img: ti.ext_arr(), flag: ti.i32):
  for i,j,k in Flags:
    if is_internal_cell(i,j,k):
      if k == nz_ext:
        ii,jj = subtract_ext(i,j)
        if Flags[i,j,k]&flag==flag:
          img[ii,jj] = 1.0


@ti.kernel
def C_to_img(img: ti.ext_arr()):
  for i,j,k in Flags:
    if is_internal_cell(i,j,k):
      ii,jj = subtract_ext(i,j)
      if k == nz_ext:
        img[ii,jj] = ti.cast(C[i,j,k],ti.f32)

@ti.kernel
def Phi_to_img(img: ti.ext_arr()):
  for i,j,k in Flags:
    if is_internal_cell(i,j,k):
      ii,jj = subtract_ext(i,j)
      if k == nz_ext:
        img[ii,jj] = ti.cast(Phi[i,j,k],ti.f32)


@ti.kernel
def M_to_img(img1: ti.ext_arr(), img2: ti.ext_arr(), img3: ti.ext_arr()):
  for i,j,k in Flags:
    if is_internal_cell(i,j,k):
      ii,jj = subtract_ext(i,j)
      if k == nz_ext:
        img1[ii,jj] = .5*ti.cast(M[i,j,k][0],ti.f32)+.5
        img2[ii,jj] = .5*ti.cast(M[i,j,k][1],ti.f32)+.5
        img3[ii,jj] = .5*ti.cast(M[i,j,k][2],ti.f32)+.5

@ti.kernel
def internal_cells_to_img(img: ti.ext_arr()):
  for i,j,k in Flags:
    if is_internal_cell(i,j,k):
      img[i,j] = 1.0

## plotting functions ##
def plot_interfaces():
  for j in range(ny_tot):
    for i in range(nx_tot):
      for k in range(nz_tot):
        if abs(M[i,j,k][0]) > 100*small or abs(M[i,j,k][1]) > 100*small:
          x,y,z = get_vert_loc(i,j,k)
          if abs(M[i,j,k][0]) > abs(M[i,j,k][1]):
            yl = np.array([y,y+dy])
            if (M[i,j,k][0] > 0.0 and M[i,j,k][1] > 0.0) or (M[i,j,k][0] < 0.0 and M[i,j,k][1] < 0.0):
              yl[0] = max(yl[0],-(M[i,j,k][0]*dx-Alpha[i,j,k])/M[i,j,k][1] + y)
              yl[1] = min(yl[1],-(M[i,j,k][0]*0.0-Alpha[i,j,k])/M[i,j,k][1] + y)
            if (M[i,j,k][0] > 0.0 and M[i,j,k][1] < 0.0) or (M[i,j,k][0] < 0.0 and M[i,j,k][1] > 0.0):
              yl[0] = max(yl[0],-(M[i,j,k][0]*0.0-Alpha[i,j,k])/M[i,j,k][1] + y)
              yl[1] = min(yl[1],-(M[i,j,k][0]*dx-Alpha[i,j,k])/M[i,j,k][1] + y)
            xl = -(M[i,j,k][1]*(yl-y) - Alpha[i,j,k])/M[i,j,k][0] + x
            plt.plot(xl,yl)
          else:
            xl = np.array([x,x+dx])
            if (M[i,j,k][0] > 0.0 and M[i,j,k][1] > 0.0) or (M[i,j,k][0] < 0.0 and M[i,j,k][1] < 0.0):
              xl[0] = max(xl[0],-(M[i,j,k][1]*dy - Alpha[i,j,k])/M[i,j,k][0] + x)
              xl[1] = min(xl[1],-(M[i,j,k][1]*0.0 - Alpha[i,j,k])/M[i,j,k][0] + x)
            if (M[i,j,k][0] > 0.0 and M[i,j,k][1] < 0.0) or (M[i,j,k][0] < 0.0 and M[i,j,k][1] > 0.0):
              xl[0] = max(xl[0],-(M[i,j,k][1]*0.0 - Alpha[i,j,k])/M[i,j,k][0] + x)
              xl[1] = min(xl[1],-(M[i,j,k][1]*dy - Alpha[i,j,k])/M[i,j,k][0] + x)
            yl = -(M[i,j,k][0]*(xl-x) - Alpha[i,j,k])/M[i,j,k][1] + y
            plt.plot(xl,yl)
  plt.grid(color='k', linestyle='-', linewidth=.25)
  plt.xticks(np.arange(0, wx, wx/nx))
  plt.yticks(np.arange(0, wy, wy/ny))
  plt.axis([0, wx, 0, wy])
  plt.show()

def write_band_png(n):
  img = np.zeros((nx,ny),dtype=np.float32)
  band_to_img(img)
  img = np.transpose(img)
  img = np.flipud(img)
  cv2.imwrite("output/band"+str(n)+".png", img * 255.0)

def write_Flag_png(n, flag, flagname):
  img = np.zeros((nx,ny),dtype=np.float32)
  Flag_to_img(img, int(flag))
  img = np.transpose(img)
  img = np.flipud(img)
  cv2.imwrite("output/"+flagname+str(n)+".png", img * 255.0)

def write_C_png(n):
  img = np.zeros((nx,ny),dtype=np.float32)+.1
  C_to_img(img)
  img = np.transpose(img)
  img = np.flipud(img)
  cv2.imwrite("output/C"+str(n)+".png", img * 255.0)
  #np.savetxt('output/C0.txt', img, fmt='%8.9f')

def write_Phi_png(n):
  img = np.zeros((nx,ny),dtype=np.float32)
  Phi_to_img(img)
  img = np.transpose(img)
  img = np.flipud(img)
  plt.contourf(img)
  plt.show()
  #cv2.imwrite("output/C"+str(n)+".png", img * 255.0)
  #np.savetxt('output/C0.txt', img, fmt='%8.9f')


def write_M_png(n):
  img1 = np.zeros((nx,ny),dtype=np.float32)+.5
  img2 = np.zeros((nx,ny),dtype=np.float32)+.5
  img3 = np.zeros((nx,ny),dtype=np.float32)+.5
  M_to_img(img1, img2, img3)
  img1 = np.transpose(img1)
  img1 = np.flipud(img1)
  img2 = np.transpose(img2)
  img2 = np.flipud(img2)
  img3 = np.transpose(img3)
  img3 = np.flipud(img3)

  #plot_interfaces()

  cv2.imwrite("output/Mx"+str(n)+".png", img1 * 255)
  cv2.imwrite("output/My"+str(n)+".png", img2 * 255)
  cv2.imwrite("output/Mz"+str(n)+".png", img3 * 255)

def show_domain():
  img = np.zeros((nx_tot,ny_tot),dtype=np.float32)
  internal_cells_to_img(img)
  plt.pcolor(img)
  plt.show()
