# -*- coding: utf-8 -*-
"""
Created on Wed May  7 10:23:42 2025

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
mpl.rcParams['font.size'] = 18
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=500 # highres display


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



# %%

# Replace 'your_file.csv' with the path to your CSV file
file_path = r"D:\data Lab\400 vs 800 doppler experiment\400 pump 400 probe\14th Feb 2024\good reflectivity\Wed Feb 14 15_56_19 2024\MeasLog.csv"

# Read the CSV file
df = pd.read_csv(file_path)

pd_signal = np.array(df["CH2 - PK2Pk"])
norm_factor = np.array(df["CH1 - PK2Pk"])
norm_signal = pd_signal/norm_factor

delay_reflectivity = np.linspace(9.5,13.5,len(pd_signal))-10.5
delay_reflectivity = 2*delay_reflectivity/c
delay_reflectivity = np.around(delay_reflectivity, decimals=3)

maxw = find_index(delay_reflectivity,4)
minw = find_index(delay_reflectivity,-2.5)
shift = 8


# %%
 
t = delay_reflectivity[minw:maxw]
sig = (norm_signal/np.max(norm_signal))[minw+shift:maxw+shift]
plt.plot(t,sig,"go--",label="reflectivity",markersize=10,lw=2)
plt.show()

# %%
from scipy.interpolate import interp1d

f_interp = interp1d(t, sig, kind='cubic')
t = np.linspace(t[0], t[-1], len(t)*2)  # 4x more samples
sig = f_interp(t)

# Centered FFT
sig_zero_mean = sig - np.mean(sig)

N = len(sig_zero_mean)
dt = t[1] - t[0]              # in picoseconds (ps), so freq is in THz

fft_vals = np.fft.fft(sig_zero_mean)
fft_vals_shifted = np.fft.fftshift(fft_vals)

freq = np.fft.fftfreq(N, d=dt)     # in THz since dt is in ps
freq = np.fft.fftshift(freq)

amplitude = np.abs(fft_vals_shifted)
# Suppress DC spike visually
dc_index = np.argmin(np.abs(freq))  # index where freq is closest to 0
amplitude[dc_index] = 1.1*amplitude[dc_index + 1]  # or average with neighbors


# %%
# --- Plotting ---
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=False)

# Plot time-domain signal
axs[0].plot(t, sig, "go--", label="Reflectivity", markersize=6, lw=1.5)
axs[0].set_ylabel("Normalized Signal")
axs[0].set_title("High resolution Time-Domain Reflectivity Signal (interpolated)")
axs[0].set_xlabel("delay (ps)")
axs[0].legend()
axs[0].grid(True)

# Plot frequency-domain signal (FFT)
axs[1].plot(freq, amplitude, "r-")
axs[1].fill_between(x=freq, y1=0, y2=amplitude, color="r", alpha=0.5)
axs[1].axvline(x=1.1, lw=2, ls="--", color="b", label="Peak = 1.1 THz")
axs[1].axvline(x=1.45, lw=2, ls="--", color="k", label="Peak = 1.45 THz")
axs[1].set_xlabel("Frequency (THz)")
axs[1].set_ylabel("Amplitude (arb unit)")
axs[1].set_title("FFT of Reflectivity Signal")
axs[1].grid(True)
axs[1].legend(loc="upper left")

plt.tight_layout()
plt.show()