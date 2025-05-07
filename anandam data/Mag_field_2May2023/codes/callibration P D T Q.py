# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:46:52 2023

@author: sagar
"""
import sys
sys.path.append("D:\\Codes\\My Python Module")
from skimage import io, draw
import matplotlib.pyplot as plt
import numpy as np
import magfield.magfield_module as mm

import matplotlib
matplotlib.rcParams["figure.dpi"]=200

filter_p = 0.86*0.5*0.27
filter_c = 0.01

image_p = io.imread("D:\\data Lab\\anandam data\\dec 2023\\Calibration P D\\P_3.tiff")
image_c = io.imread("D:\\data Lab\\anandam data\\dec 2023\\Calibration T Q\\C_3.tiff")


def get_image_tl_and_br(image_p):
    # Define the vertices of the top-left triangle
    triangle_vertices = np.array([[0, 0], [0, image_p.shape[1]-1], [image_p.shape[0]-1, 0]])
    
    # Create a binary mask for the triangle region
    mask = np.zeros_like(image_p, dtype=bool)
    rr, cc = draw.polygon(triangle_vertices[:, 0], triangle_vertices[:, 1])
    mask[rr, cc] = True
    
    # Apply the mask to the original image
    image_p_tl = np.zeros_like(image_p)
    image_p_tl[mask] = image_p[mask]
    # Replace NaN values with zero
    image_p_tl = np.nan_to_num(image_p_tl, nan=0)
    image_p_agv = 1.2*np.mean(image_p[0:400,600:1000])
    image_p_tl = image_p_tl*(image_p_tl>image_p_agv)
    
    # Define the vertices of the top-left triangle
    triangle_vertices = np.array([[image_p.shape[0]-1, image_p.shape[1]-1], [image_p.shape[0]-1, 0], [0, image_p.shape[1]-1]])
    
    # Create a binary mask for the triangle region
    mask = np.zeros_like(image_p, dtype=bool)
    rr, cc = draw.polygon(triangle_vertices[:, 0], triangle_vertices[:, 1])
    mask[rr, cc] = True
    
    # Apply the mask to the original image
    image_p_br = np.zeros_like(image_p)
    image_p_br[mask] = image_p[mask]
    # Replace NaN values with zero
    image_p_br = np.nan_to_num(image_p_br, nan=0)
    image_p_br = image_p_br*(image_p_br>image_p_agv)
    
    return image_p_tl, image_p_br

def get_nonzero_value_average(matrix):
    # Extract the nonzero elements
    nonzero_elements = matrix[matrix != 0.0]

    # Calculate the average of nonzero elements
    average_nonzero = np.mean(nonzero_elements)
    return(average_nonzero)

image_p_tl, image_p_br = get_image_tl_and_br(image_p)

image_p_avg = np.mean(image_p[0:400,600:1000])
print(f"Image p background: {image_p_avg}")

image_p_tl_avg = get_nonzero_value_average(image_p_tl)
image_p_br_avg = get_nonzero_value_average(image_p_br)

cal_p_tl = (image_p_tl_avg-image_p_avg)/filter_p
cal_p_br = (image_p_br_avg-image_p_avg)/filter_p



## Cross calculation 
#
#
###############################

image_c_tl, image_c_br = get_image_tl_and_br(image_c)

image_c_avg = np.mean(image_c[0:400,600:1000])
print(f"Image c background: {image_c_avg}")

image_c_tl_avg = get_nonzero_value_average(image_c_tl)
image_c_br_avg = get_nonzero_value_average(image_c_br)

cal_c_tl = (image_c_tl_avg-image_c_avg)/filter_c
cal_c_br = (image_c_br_avg-image_c_avg)/filter_c

D = cal_p_tl
P = cal_p_br

Q = cal_c_tl
T = cal_c_br

print(f"callibration factor for P:  {P/Q}")
print(f"callibration factor for D:  {D/Q}")
print(f"callibration factor for Q:  {Q/Q}")
print(f"callibration factor for T:  {T/Q}")




plt.imshow(image_p_tl, cmap="jet")
plt.title("parallel top left (D)")
plt.xlabel(f"callibration factor for D:  {D/Q}")
plt.colorbar()
plt.show()

plt.imshow(image_p_br, cmap="jet")
plt.title("parallel bottom right (P)")
plt.xlabel(f"callibration factor for P:  {P/Q}")
plt.colorbar()
plt.show()

plt.imshow(image_c_tl, cmap="jet")
plt.title("cross top left (Q)")
plt.xlabel(f"callibration factor for Q:  {Q/Q}")
plt.colorbar()
plt.show()

plt.imshow(image_c_br, cmap="jet")
plt.title("cross bottom right (T)")
plt.xlabel(f"callibration factor for T:  {T/Q}")
plt.colorbar()
plt.show()


pump_image_p = io.imread("D:\\data Lab\\anandam data\\dec 2023\\27th dec 2023\\27th dec 2023 P D\\pump only\\P_003.tif")


pump_image_p_tl, pump_image_p_br = mm.get_image_tl_and_br(image=pump_image_p, background_factor=1.2)
plt.imshow(pump_image_p_tl)
plt.title("D")
plt.colorbar()
plt.show()

plt.imshow(pump_image_p_br)
plt.title("P")
plt.colorbar()
plt.show()

d_image_pump_noise = mm.get_nonzero_value_average(pump_image_p_tl)
p_image_pump_noise = mm.get_nonzero_value_average(pump_image_p_br)


pump_image_c = io.imread("D:\\data Lab\\anandam data\\dec 2023\\27th dec 2023\\27th dec 2023 T Q\\pump only\\C_003.tif")


pump_image_c_tl, pump_image_c_br = mm.get_image_tl_and_br(image=pump_image_c, background_factor=1.02)
plt.imshow(pump_image_c_tl)
plt.title("Q")
plt.colorbar()
plt.show()

plt.imshow(pump_image_c_br)
plt.title("T")
plt.colorbar()
plt.show()

q_image_pump_noise = mm.get_nonzero_value_average(pump_image_c_tl)
t_image_pump_noise = mm.get_nonzero_value_average(pump_image_c_br)

def subtract_pump_noise(matrix, noise):
    noise = np.uint16(noise)
    # Convert the matrix to a NumPy array with the specified data type
    matrix_array = np.array(matrix)
    
    # Replace negative values with zero
    matrix_array[matrix_array > noise] -= noise
    
    return matrix_array.tolist()  # Convert the NumPy array back to a list


image = io.imread("D:\\data Lab\\anandam data\\dec 2023\\27th dec 2023\\27th dec 2023 P D\\pump probe\\P_045.tif")

D,P = mm.get_image_tl_and_br(image, background_factor=1.1)

# D -= np.uint16(d_image_pump_noise)
# P -= np.uint16(p_image_pump_noise)

D=subtract_pump_noise(D, d_image_pump_noise)
P=subtract_pump_noise(P, p_image_pump_noise)

image = io.imread("D:\\data Lab\\anandam data\\dec 2023\\27th dec 2023\\27th dec 2023 T Q\\pump probe\\C_045.tif")

Q,T = mm.get_image_tl_and_br(image, background_factor=1.1)

# Q -= np.uint16(q_image_pump_noise)
# T -= np.uint16(t_image_pump_noise)

Q=subtract_pump_noise(Q, q_image_pump_noise)
T=subtract_pump_noise(T, t_image_pump_noise)


plt.imshow(D)
plt.title("D")
plt.show()

plt.imshow(P)
plt.title("P")
plt.show()

plt.imshow(Q)
plt.title("Q")
plt.show()

plt.imshow(T)
plt.title("T")
plt.show()

print(P)