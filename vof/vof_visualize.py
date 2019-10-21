import numpy as np
import matplotlib.pyplot as plt
import cv2
from vof_data import *

@ti.kernel
def Flags_to_img(img: np.ndarray):
	for i,j,k in Flags:
		if k == 1:
			if Flags[i,j,k]&cell_flags.CELL_INTERFACE==cell_flags.CELL_INTERFACE:
				img[i+ n_x*j] = 1.0
			elif Flags[i,j,k]&cell_flags.CELL_ACTIVE==cell_flags.CELL_ACTIVE:
				img[i+ n_x*j] = 2.0/3.0
			elif Flags[i,j,k]&cell_flags.CELL_GHOST==cell_flags.CELL_GHOST:
				img[i+ n_x*j] = 1.0/3.0
			else:
				img[i+ n_x*j] = 0.0

@ti.kernel
def C_to_img(img: np.ndarray):
	for i,j,k in Flags:
		if k == 1:
			img[i+ n_x*j] = ti.cast(C[i,j,k],ti.f32)

@ti.kernel
def M_to_img(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray):
	for i,j,k in Flags:
		if k == 1:
			img1[i+ n_x*j] = ti.abs(ti.cast(M[i,j,k][0],ti.f32))
			img2[i+ n_x*j] = ti.abs(ti.cast(M[i,j,k][1],ti.f32))
			img3[i+ n_x*j] = ti.abs(ti.cast(M[i,j,k][2],ti.f32))

def write_Flags_png(n):
	img = np.zeros((n_x*n_y),dtype=np.float32)+.1
	Flags_to_img(img)
	img = img.reshape(n_x,n_y)
	cv2.imwrite("output/Flags"+str(n)+".png", img * 255)

def write_C_png(n):
	img = np.zeros((n_x*n_y),dtype=np.float32)
	C_to_img(img)
	img = img.reshape(n_x,n_y)
	cv2.imwrite("output/C"+str(n)+".png", img * 255)
	#np.savetxt('C0.txt', img, fmt='%8.9f')

def write_M_png(n):
	img1 = np.zeros((n_x*n_y),dtype=np.float32)
	img2 = np.zeros((n_x*n_y),dtype=np.float32)
	img3 = np.zeros((n_x*n_y),dtype=np.float32)
	M_to_img(img1, img2, img3)
	img1 = img1.reshape(n_x,n_y)
	img2 = img2.reshape(n_x,n_y)
	img3 = img3.reshape(n_x,n_y)
	cv2.imwrite("output/Mx"+str(n)+".png", img1 * 255)
	cv2.imwrite("output/My"+str(n)+".png", img2 * 255)
	cv2.imwrite("output/Mz"+str(n)+".png", img3 * 255)
