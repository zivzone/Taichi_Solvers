import numpy as np
import matplotlib.pyplot as plt
import cv2
from vof_data import *
from vof_common import *

@ti.func
def ij_to_img_idx(i,j):
  idx = i-nx_ghost + nx*(j-ny_ghost)
  return idx

@ti.kernel
def Flags_to_img(img: np.ndarray):
  for i,j,k in Flags:
    if is_internal_cell(i,j,k):
      if k == nz_ghost+6:
        idx = ij_to_img_idx(i,j)
        if is_interface_cell(i,j,k):
          img[idx] = 1.0
        elif is_active_cell(i,j,k):
          img[idx] = 2.0/3.0
        elif is_ghost_cell(i,j,k):
          img[idx] = 1.0/3.0
        else:
          img[idx] = 0.0


@ti.kernel
def C_to_img(img: np.ndarray):
  for i,j,k in Flags:
    if is_internal_cell(i,j,k):
      if k == nz_ghost+1:
        idx = ij_to_img_idx(i,j)
        img[idx] = ti.cast(C[i,j,k],ti.f32)


@ti.kernel
def M_to_img(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray):
  for i,j,k in Flags:
    if is_internal_cell(i,j,k):
      if k == nz_ghost+1:
        idx = ij_to_img_idx(i,j)
        img1[idx] = ti.abs(ti.cast(M[i,j,k][0],ti.f32))
        img2[idx] = ti.abs(ti.cast(M[i,j,k][1],ti.f32))
        img3[idx] = ti.abs(ti.cast(M[i,j,k][2],ti.f32))


def write_Flags_png(n):
  img = np.zeros((nx*ny),dtype=np.float32)+.1
  Flags_to_img(img)
  img = img.reshape(nx,ny)
  cv2.imwrite("output/Flags"+str(n)+".png", img * 255)


def write_C_png(n):
  img = np.zeros((nx*ny),dtype=np.float32)+.1
  C_to_img(img)
  img = img.reshape(nx,ny)
  cv2.imwrite("output/C"+str(n)+".png", img * 255)
  #np.savetxt('C0.txt', img, fmt='%8.9f')


def write_M_png(n):
  img1 = np.zeros((nx*ny),dtype=np.float32)
  img2 = np.zeros((nx*ny),dtype=np.float32)
  img3 = np.zeros((nx*ny),dtype=np.float32)
  M_to_img(img1, img2, img3)
  img1 = img1.reshape(nx,ny)
  img2 = img2.reshape(nx,ny)
  img3 = img3.reshape(nx,ny)
  cv2.imwrite("output/Mx"+str(n)+".png", img1 * 255)
  cv2.imwrite("output/My"+str(n)+".png", img2 * 255)
  cv2.imwrite("output/Mz"+str(n)+".png", img3 * 255)
