import numpy as np
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
				img[i+ n_x*j] = 1.0/3.

@ti.kernel
def C_to_img(img: np.ndarray):
	for i,j,k in Flags:
		if k == 1:
			img[i+ n_x*j] = C[i,j,k]

@ti.kernel
def M_to_img(img: np.ndarray):
	for i,j,k in Flags:
		if k == 1:
			img[i+ n_x*j] = M[i,j,k][0]

def write_Flags_png(file_name):
	img = np.zeros((n_x*n_y),dtype=np.float32)+.1
	Flags_to_img(img)
	img = img.reshape(n_x,n_y)
	cv2.imwrite(file_name, img * 255)

def write_C_png(file_name):
	img = np.zeros((n_x*n_y),dtype=np.float32)+.1
	Flags_to_img(img)
	img = img.reshape(n_x,n_y)
	cv2.imwrite(file_name, img * 255)

def write_M_png(file_name):
	img = np.zeros((n_x*n_y),dtype=np.float32)+.1
	Flags_to_img(img)
	img = img.reshape(n_x,n_y)
	cv2.imwrite(file_name, img * 255)
