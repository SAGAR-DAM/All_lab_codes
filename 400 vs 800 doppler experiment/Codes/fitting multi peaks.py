# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:24:13 2024
@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from Curve_fitting_with_scipy import Multi_Gaussianfitting as mgf
#from Curve_fitting_with_scipy import Gaussianfitting as Gf
from Curve_fitting_with_scipy import Linefitting as lft
#from Curve_fitting_with_scipy import Lorentzianfitting as loft
from scipy.signal import fftconvolve
from scipy.interpolate import CubicSpline

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

c = 0.3   #in mm/ps


def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index

def double_density_interpolation(x, y, x_range):
    # Perform cubic interpolation
    cs = CubicSpline(x, y)
    
    # Double the number of points in the x-axis
    x_interp = np.linspace(min(x_range), max(x_range), len(x_range) * 2)
    
    # Perform cubic interpolation on the new x points
    y_interp = cs(x_interp)
    
    return x_interp, y_interp

def moving_average(signal, window_size):
    # Define the window coefficients for the moving average
    window = np.ones(window_size) / float(window_size)
    
    # Apply the moving average filter using fftconvolve
    filtered_signal = fftconvolve(signal, window, mode='same')
    
    return filtered_signal

files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14th Feb 2024\\Spectrum\\run7\\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run9_70%_20TW_ret_11-15_250fs\\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run8_30%_20TW_ret_11-15_250fs\\*.txt")

delay = np.linspace(9.5,13.5,len(files)//2)-10.5
delay = 2*delay/c
delay = np.around(delay, decimals=3)


filenumber = 9
f = open(files[filenumber])
r=np.loadtxt(f,skiprows=17,comments='>')

wavelength = r[:,0]
intensity = r[:,1]
intensity /= max(intensity)


minw = find_index(wavelength, 390)
maxw = find_index(wavelength, 420)

w = wavelength[minw:maxw]
I = intensity[minw:maxw]

line, parameters = lft.linefit(w,I)
I1 = ((I-line)>0)*(I-line)
I1 = moving_average(I1, 5)

minw = find_index(w,400)
maxw = find_index(w,418)

fitting_order = 7
multiplication_factor = 10
fit_y,parameters = mgf.Multi_Gaussfit(multiplication_factor*w[minw:maxw],I1[minw:maxw],fitting_order)

for i in range(fitting_order):
    parameters[3*i+1] = float(parameters[3*i+1])/multiplication_factor
    parameters[3*i+2] = float(parameters[3*i+2])/multiplication_factor

plt.figure()
plt.plot(w,I-line,'ko',label="data")
plt.plot(w[minw:maxw],I1[minw:maxw],'g-', label = "Modified I")
plt.plot(w[minw:maxw],fit_y,'r-',lw=2, label="fit")
for i in range(len(parameters)//3):
    if(w[0]<=float(parameters[3*i+2])<=w[-1]):
        plt.text(float(parameters[3*i+2]), 1.08*float(parameters[3*i]), f'{1+i}', fontsize=12, ha='center', va='bottom', color="blue")
plt.figtext(0.95,0.2,str(parameters), fontsize=9)
plt.legend()
plt.xlim(w[minw],w[maxw])
plt.ylim(1.1*min([min((I-line)[minw:maxw]),min(I1[minw:maxw]),min(fit_y)]),1.15*max([max((I-line)[minw:maxw]),max(I1[minw:maxw]),max(fit_y)]))
plt.grid(lw=0.5, color="black")
plt.show()  
