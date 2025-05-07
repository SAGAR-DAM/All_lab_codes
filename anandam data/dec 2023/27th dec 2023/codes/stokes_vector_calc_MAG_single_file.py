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
import glob

import matplotlib
matplotlib.rcParams["figure.dpi"]=100

#
filter_p = 0.86*0.5*0.27
filter_c = 0.01

image_p = io.imread("D:\\Data Lab\Polarimetry\\dec 2023\\27th dec\\Calibration P D\\P_3.tiff")
image_c = io.imread("D:\\Data Lab\Polarimetry\\dec 2023\\27th dec\\Calibration T Q\\C_3.tiff")



image_p_tl, image_p_br = mm.get_image_tl_and_br(image=image_p, background_factor=1.2)
image_c_tl, image_c_br = mm.get_image_tl_and_br(image=image_c, background_factor=1.2)

##
## Calculation for parallel arm
##

image_p_avg = np.mean(image_p[0:400,600:1000])
print(f"Image p background: {image_p_avg}")

image_p_tl_avg = mm.get_nonzero_value_average(image_p_tl)
image_p_br_avg = mm.get_nonzero_value_average(image_p_br)

cal_p_tl = (image_p_tl_avg-image_p_avg)/filter_p
cal_p_br = (image_p_br_avg-image_p_avg)/filter_p


##
## Calculation for crossed arm
##

image_c_avg = np.mean(image_c[0:400,600:1000])
print(f"Image c background: {image_c_avg}")

image_c_tl_avg = mm.get_nonzero_value_average(image_c_tl)
image_c_br_avg = mm.get_nonzero_value_average(image_c_br)

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


#

# plt.imshow(image_p_tl, cmap="jet")
# plt.title("parallel top left (D)")
# plt.xlabel(f"callibration factor for D:  {D/Q}")
# plt.colorbar()
# plt.show()

# plt.imshow(image_p_br, cmap="jet")
# plt.title("parallel bottom right (P)")
# plt.xlabel(f"callibration factor for P:  {P/Q}")
# plt.colorbar()
# plt.show()

# plt.imshow(image_c_tl, cmap="jet")
# plt.title("cross top left (Q)")
# plt.xlabel(f"callibration factor for Q:  {Q/Q}")
# plt.colorbar()
# plt.show()

# plt.imshow(image_c_br, cmap="jet")
# plt.title("cross bottom right (T)")
# plt.xlabel(f"callibration factor for T:  {T/Q}")
# plt.colorbar()
# plt.show()


#
pump_image_p = io.imread("D:\\Data Lab\\Polarimetry\\dec 2023\\27th dec\\27th dec 2023 P D\\Magnet\\P_006.tif")


pump_image_p_tl, pump_image_p_br = mm.get_image_tl_and_br(image=pump_image_p, background_factor=1.0)
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

print(f"D pump noise: {d_image_pump_noise}")
print(f"P pump noise: {p_image_pump_noise}")
pump_image_c = io.imread("D:\\Data Lab\\Polarimetry\\dec 2023\\27th dec\\27th dec 2023 T Q\\Magnet\\C_006.tif")


pump_image_c_tl, pump_image_c_br = mm.get_image_tl_and_br(image=pump_image_c, background_factor=1.0)
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

print(f"Q pump noise: {q_image_pump_noise}")
print(f"T pump noise: {t_image_pump_noise}")

#
def subtract_pump_noise(matrix, noise):
    noise = np.uint16(noise)
    # Convert the matrix to a NumPy array with the specified data type
    matrix_array = np.array(matrix)
    
    # Replace negative values with zero
    matrix_array[matrix_array > noise] -= noise
    
    return matrix_array.tolist()  # Convert the NumPy array back to a list



def get_stokes(image_P,image_C, background_factor):
    D,P = mm.get_image_tl_and_br(image_P, background_factor)
    
    # D -= np.uint16(d_image_pump_noise)
    # P -= np.uint16(p_image_pump_noise)
    
    D=subtract_pump_noise(D, d_image_pump_noise)
    P=subtract_pump_noise(P, p_image_pump_noise)
    
    
    
    Q,T = mm.get_image_tl_and_br(image_C, background_factor)
    
    # Q -= np.uint16(q_image_pump_noise)
    # T -= np.uint16(t_image_pump_noise)
    
    Q=subtract_pump_noise(Q, q_image_pump_noise)
    T=subtract_pump_noise(T, t_image_pump_noise)
    
    
    # plt.imshow(D)
    # plt.title("D")
    # plt.colorbar()
    # plt.show()
    
    # plt.imshow(P)
    # plt.title("P")
    # plt.colorbar()
    # plt.show()
    
    # plt.imshow(Q)
    # plt.title("Q")
    # plt.colorbar()
    # plt.show()
    
    # plt.imshow(T)
    # plt.title("T")
    # plt.colorbar()
    # plt.show()
    

    p_filter = 792.82
    d_filter = 415.2
    t_filter = 771.51
    q_filter = 333.0
    
    P_int = np.sum(P)*p_filter
    D_int = np.sum(D)*d_filter
    Q_int = np.sum(Q)*q_filter
    T_int = np.sum(T)*t_filter
    
    
    s1 = 2*P_int/D_int-1
    s2 = 2*T_int/D_int-1
    s3 = 1-2*Q_int/D_int
    
    s = np.matrix([[s1],[s2],[s3]])
    mod_s = np.sqrt(np.dot(s.T,s)[0,0])
    
    return s, mod_s

def get_mod_s(image_P,image_C, background_factor):
    D,P = mm.get_image_tl_and_br(image_P, background_factor)
    
    # D -= np.uint16(d_image_pump_noise)
    # P -= np.uint16(p_image_pump_noise)
    
    D=subtract_pump_noise(D, d_image_pump_noise)
    P=subtract_pump_noise(P, p_image_pump_noise)
    
    
    
    Q,T = mm.get_image_tl_and_br(image_C, background_factor)
    
    # Q -= np.uint16(q_image_pump_noise)
    # T -= np.uint16(t_image_pump_noise)
    
    Q=subtract_pump_noise(Q, q_image_pump_noise)
    T=subtract_pump_noise(T, t_image_pump_noise)

    p_filter = 792.82
    d_filter = 415.2
    t_filter = 771.51
    q_filter = 333.0
    
    P_int = np.sum(P)*p_filter
    D_int = np.sum(D)*d_filter
    Q_int = np.sum(Q)*q_filter
    T_int = np.sum(T)*t_filter
    
    
    s1 = 2*P_int/D_int-1
    s2 = 2*T_int/D_int-1
    s3 = 1-2*Q_int/D_int
    
    s = np.matrix([[s1],[s2],[s3]])
    mod_s = np.sqrt(np.dot(s.T,s)[0,0])
    
    return(mod_s-1)

def bisection(f, a, b, image_P, image_C, tol, max_iter=1000):
    """
    Bisection method to find the root of the function f(x) within the interval [a, b].

    Parameters:
    - f: The function for which to find the root.
    - a: The lower limit of the interval.
    - b: The upper limit of the interval.
    - tol: Tolerance for the error. Default is 1e-6.
    - max_iter: Maximum number of iterations. Default is 1000.

    Returns:
    - root: Approximate root of the function within the specified tolerance.
    - iterations: Number of iterations performed.
    """

    if f(image_P, image_C, a) * f(image_P, image_C, b) > 0:
        raise ValueError("The function values at the interval endpoints must have different signs.")

    root = None
    iterations = 0

    while iterations < max_iter:
        c = (a + b) / 2
        if abs(f(image_P, image_C, a)) <= tol:
            root = a
            break
        elif abs(f(image_P, image_C, b)) <= tol:
            root = b
            break
        elif abs(f(image_P, image_C, c)) <= tol:
            root = c
            break
        elif f(image_P, image_C, c) * f(image_P, image_C, a) < 0:
            b = c
        else:
            a = c

        iterations += 1

    #root = (a + b) / 2
    return root


def find_interval_with_sign_change(f, lower, upper, image_P, image_C, step=0.1):
    """
    Find an interval [a, b] such that f(a) * f(b) < 0.

    Parameters:
    - f: The function for which to find the interval.
    - lower: Lower limit of the range.
    - upper: Upper limit of the range.
    - step: Step size for iterating over the range. Default is 0.1.

    Returns:
    - (a, b): Interval [a, b] where f(a) * f(b) < 0.
    """

    a = None
    b = None

    x = lower
    while x < upper:
        if f(image_P, image_C, x) * f(image_P, image_C, x + step) < 0:
            a = x
            b = x + step
            break
        x += step

    return a, b


def make_s_normalized(image_P, image_C):
    try:
        background_factor = bisection(f = get_mod_s, a=1, b=1.2, image_P=image_P, image_C=image_C, tol=0.01)
    except:
        a, b = find_interval_with_sign_change(f = get_mod_s, lower = 1, upper = 2, image_P= image_P, image_C=image_C)
        background_factor = bisection(f = get_mod_s, a=a, b=b, image_P=image_P, image_C=image_C, tol=0.01)
    return(background_factor)
    



files_p = glob.glob("D:\\Data Lab\\Polarimetry\\dec 2023\\27th dec\\27th dec 2023 P D\\Magnet\\*.tif")
files_c = glob.glob("D:\\Data Lab\\Polarimetry\\dec 2023\\27th dec\\27th dec 2023 T Q\\Magnet\\*.tif")


successful_files = 0
bg_factor_array = []

for i in range(5,len(files_p)):
    image_P = io.imread(files_p[i])
    image_C = io.imread(files_c[i])
    try:
        background_factor = make_s_normalized(image_P, image_C)
        
        s,mod_s=get_stokes(image_P, image_C, background_factor)
        
        print("\n\n"+f"file:  {files_p[i][-7:]}")
        print(f"s  {s}")
        print(f"|s|:  {mod_s}")
        print("|mod_s - 1|:  ", abs(mod_s-1))
        print(f"background factor:  {background_factor}")
        
        print("\n#########################################")
        
        successful_files += 1
        bg_factor_array.append(background_factor)
        
    except:
        None
        
        
print(f"successful_files:  {successful_files}")

plt.plot(bg_factor_array)
plt.ylim(1,2)
plt.show()