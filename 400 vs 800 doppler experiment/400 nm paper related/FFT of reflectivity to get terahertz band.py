# -*- coding: utf-8 -*-
"""
Created on Wed May  7 11:28:05 2025

@author: mrsag
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:24:13 2024

@author: mrsag
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import glob
from Curve_fitting_with_scipy import Gaussianfitting as Gf
from scipy.signal import fftconvolve
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
import pandas as pd 

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=500 # highres display


# %%


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
    filtered_signal = fftconvolve(signal, window, mode='same')
    
    return filtered_signal


# %%

c = 0.3
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
    filtered_signal = fftconvolve(signal, window, mode='same')
    
    return filtered_signal


files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14th Feb 2024\\Spectrum\\run7\\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run9_70%_20TW_ret_11-15_250fs\\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run8_30%_20TW_ret_11-15_250fs\\*.txt")

delay = np.linspace(9.5,13.5,len(files)//2)-10.5
delay = 2*delay/c
delay = np.around(delay, decimals=3)

peaks = []

# for i in range(1,len(files),2):
#     f = open(files[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     minw = find_index(wavelength, 392)
#     maxw = find_index(wavelength, 400)
    
#     wavelength = wavelength[minw:maxw]
#     intensity = intensity[minw:maxw] 

#     intensity -= np.mean(intensity[0:50])
    
#     fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
#     peaks.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i]}")
#     #plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

for i in range(1,len(files),2):
    f = open(files[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    
    
    minw = find_index(wavelength, 390)
    maxw = find_index(wavelength, 400)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    intensity -= np.mean(intensity[0:50])
    intensity /= max(intensity)
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks.append(parameters[1])
    
    # plt.plot(wavelength, intensity, '-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks)):
    if (peaks[i]<395 or peaks[i]>396):
        try:
            peaks[i] = (peaks[i+1]+peaks[i-1])/2
        except:
            peaks[i] = 395.35

peaks = moving_average(peaks,1)

for i in range(len(peaks)):
    if (peaks[i]<395 or peaks[i]>396):
        try:
            if((peaks[i-1]<395 and peaks[i+1]>396) or (peaks[i+1]<395 and peaks[i+1]>396)):
                peaks[i] = (peaks[i+1]+peaks[i-1])/2
            else:
                peaks[i] = 395.35
        except:
            peaks[i] = 395.35

t = delay[0:len(peaks)]
sig = peaks-395.3

# %%

# from scipy.interpolate import interp1d

# f_interp = interp1d(t, sig, kind='cubic')
# t = np.linspace(t[0], t[-1], len(t)*2)  # 4x more samples
# sig = f_interp(t)

# Centered FFT
sig_zero_mean = sig #- np.mean(sig)

N = len(sig_zero_mean)
dt = t[1] - t[0]              # in picoseconds (ps), so freq is in THz

fft_vals = np.fft.fft(sig_zero_mean)
fft_vals_shifted = np.fft.fftshift(fft_vals)

freq = np.fft.fftfreq(N, d=dt)     # in THz since dt is in ps
freq = np.fft.fftshift(freq)

amplitude = np.abs(fft_vals_shifted)
# Suppress DC spike visually
# dc_index = np.argmin(np.abs(freq))  # index where freq is closest to 0
# amplitude[dc_index] = 1.1*amplitude[dc_index + 1]  # or average with neighbors


# %%
# --- Plotting ---
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=False)

# Plot time-domain signal
axs[0].plot(t, sig, "ko--", label="doppler shift", markersize=6, lw=1.5)
axs[0].set_ylabel("Normalized Signal")
axs[0].set_title("Time-Domain Reflectivity Signal (interpolated)")
axs[0].set_xlabel("time (ps)")
axs[0].legend()
axs[0].grid(True)

# Plot frequency-domain signal (FFT)
axs[1].plot(freq, amplitude, "r-")
axs[1].fill_between(x=freq, y1=0, y2=amplitude, color="r", alpha=0.5)
# axs[1].axvline(x=1.1, lw=2, ls="--", color="b", label="Peak = 1.1 THz")
axs[1].axvline(x=1.4, lw=2, ls="--", color="k", label="Peak = 1.45 THz")
axs[1].set_xlabel("Frequency (THz)")
axs[1].set_ylabel("Amplitude (arb unit)")
axs[1].set_title("FFT of Reflectivity Signal")
axs[1].grid(True)
axs[1].legend(loc="upper left")

plt.tight_layout()
plt.show()