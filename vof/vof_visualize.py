import numpy as np
import matplotlib.pyplot as plt
import cv2
from vof_data import *

@ti.kernel
def Flags_to_img(img: np.ndarray):
	for i,j,k in Flags:
		if k == 1:
			if Flags[i,j,k]&cellFlags.CELL_INTERFACE==cellFlags.CELL_INTERFACE:
				img[i+ n_x*j] = 1.0
			elif Flags[i,j,k]&cellFlags.CELL_ACTIVE==cellFlags.CELL_ACTIVE:
				img[i+ n_x*j] = 2.0/3.0
			elif Flags[i,j,k]&cellFlags.CELL_GHOST==cellFlags.CELL_GHOST:
				img[i+ n_x*j] = 1.0/3.0

@ti.kernel
def C_to_img(img: np.ndarray):
	for i,j,k in Flags:
		if k == 1:
			img[i+ n_x*j] = ti.cast(C[i,j,k],ti.f32)

@ti.kernel
def M_to_img(img: np.ndarray):
	for i,j,k in Flags:
		if k == 1:
			img[i+ n_x*j] = ti.cast(M[i,j,k][0],ti.f32)

def write_Flags_png(file_name):
	img = np.zeros((n_x*n_y),dtype=np.float32)+.1
	Flags_to_img(img)
	img = img.reshape(n_x,n_y)
	cv2.imwrite(file_name, img * 255)

def write_C_png(file_name):
	img = np.zeros((n_x*n_y),dtype=np.float32)
	C_to_img(img)
	img = img.reshape(n_x,n_y)
	cv2.imwrite(file_name, img * 255)
	#np.savetxt('C0.txt', img, fmt='%8.9f')

def write_M_png(file_name):
	img = np.zeros((n_x*n_y),dtype=np.float32)+.1
	M_to_img(img)
	img = img.reshape(n_x,n_y)
	cv2.imwrite(file_name, img * 255)
