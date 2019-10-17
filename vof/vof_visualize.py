import numpy as np
import cv2
from vof_data import *


@ti.kernel
def copy_to_img(img: np.ndarray):
	for i,j,k in Flags:
		if k == 0:
			img[i+ n_x*j] = C[i,j,k] #Flags[i,j,k]

def write_png():
	img = np.zeros((n_x*n_y),dtype=np.float32)+.1
	copy_to_img(img)
	img = img.reshape(n_x,n_y)
	cv2.imwrite('slice0.png', img * 255)
