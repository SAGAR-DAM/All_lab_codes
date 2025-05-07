# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:24:13 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from Curve_fitting_with_scipy import Gaussianfitting as Gf
from scipy.signal import fftconvolve
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize
import random

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 8
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display




c = 0.3   #in mm/ps
min_wavelength=400
max_wavelength=420

def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index


def moving_average(signal, window_size):
    # Define the window coefficients for the moving average
    window = np.ones(window_size) / float(window_size)
    
    # Apply the moving average filter using fftconvolve
    filtered_signal = fftconvolve(signal, window, mode='valid')
    
    return filtered_signal




filename = "D:\\data Lab\\400 vs 800 doppler experiment\\ameya_sol_try\\diode_2_88.txt"

f = open(filename)
r=np.loadtxt(f,skiprows=1,comments='>')

wavelength = r[:,1]
intensity = r[:,2]


intensity -= np.mean(intensity[0:200])
intensity /= max(intensity)

minw = find_index(wavelength, min(wavelength))
maxw = find_index(wavelength, max(wavelength))

wavelength = wavelength[minw:maxw]
intensity = intensity[minw:maxw] 



plt.plot(wavelength, intensity)
plt.title(f"range: {wavelength[-1]-wavelength[0]}")
plt.show()





filename = "D:\\data Lab\\400 vs 800 doppler experiment\\ameya_sol_try\\diode_1_93.txt"

f = open(filename)
r=np.loadtxt(f,skiprows=1,comments='>')

wavelength = r[:,1]
intensity = r[:,2]


intensity -= np.mean(intensity[0:200])
intensity /= max(intensity)

minw = find_index(wavelength, min(wavelength))
maxw = find_index(wavelength, max(wavelength))

wavelength = wavelength[minw:maxw]
intensity = intensity[minw:maxw] 



plt.plot(wavelength, intensity)
plt.title(f"range: {wavelength[-1]-wavelength[0]}")
plt.show()




