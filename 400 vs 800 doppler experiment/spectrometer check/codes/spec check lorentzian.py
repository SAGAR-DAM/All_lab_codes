# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:47:31 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from Curve_fitting_with_scipy import Gaussianfitting as Gf
from Curve_fitting_with_scipy import Lorentzianfitting as Lf
from scipy.signal import fftconvolve

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=100 # highres display



# R² calculation function
def calculate_r_squared(y, y_fit):
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


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



files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\spectrometer check\\HR 4000 with diode\\*.txt")
min_wavelength = 400
max_wavelength = 407

w0_arr = []
r_sq_arr = []

for file in files[:30]:
    f = open(file)
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    
    if(max(intensity)>200):
        intensity -= np.mean(intensity[0:200])
        intensity /= max(intensity)
        
        minw = find_index(wavelength, min_wavelength)
        maxw = find_index(wavelength, max_wavelength)
        
        wavelength = wavelength[minw:maxw]
        intensity = intensity[minw:maxw]
        
        try:
            fit_I,parameters,string = Lf.Lorentzfit(wavelength, intensity)
            r_sq = calculate_r_squared(intensity,fit_I)
            
            if(r_sq>=0.9):
                w0_arr.append(parameters[2])
                r_sq_arr.append(r_sq)
                plt.plot(wavelength, intensity, 'r-', label = "raw data")
                plt.plot(wavelength, fit_I, 'b-', label = " Lorentz fit")
                plt.title(f"Probe only {file[-16:-3]}")
                # plt.xlim(390,420)
                plt.xlabel("Wavelength (nm)\n"+f"Peak: {parameters[2]:.3f} nm ; r_sq_fit: {r_sq:.3f}")
                plt.ylabel("Intensity (arb unit)")
                plt.show()   
        except:
            pass
        
trendline = moving_average(signal=w0_arr, window_size=30)

plt.plot(w0_arr,'k-',lw=0.5,label="peak vals")
plt.plot(trendline,'r-',lw=1,label="trendline")
plt.title("Zitter of peak fit")
plt.xlabel("Shot no"+f"\nstd: {np.std(w0_arr):.3f} nm;  max-min: {(np.max(w0_arr)-np.min(w0_arr)):.3f} nm; trendline range: {(np.max(trendline)-np.min(trendline)):.3f} nm")
plt.ylabel("peak wavelength (nm)")
plt.show()

plt.plot(r_sq_arr,'b-',lw=0.5)
plt.title("fitting accuraacy")
plt.xlabel("Shot no"+f"\nMean r_sq = {np.mean(r_sq_arr):.3f}")
plt.ylabel("r_sq_error (dimensionless)")
plt.show()