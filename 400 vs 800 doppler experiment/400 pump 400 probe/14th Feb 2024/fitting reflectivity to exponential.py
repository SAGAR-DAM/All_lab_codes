# -*- coding: utf-8 -*-
"""
Created on Sat May  3 11:34:40 2025

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from Curve_fitting_with_scipy import Gaussianfitting as Gf
from scipy.signal import fftconvolve
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from Curve_fitting_with_scipy import exp_decay_fit as eft
import pandas as pd

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=500 # highres display

# %%
c = 0.3 # in nm/ps

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

file_path = r"D:\data Lab\400 vs 800 doppler experiment\400 pump 400 probe\14th Feb 2024\good reflectivity\Wed Feb 14 15_56_19 2024\MeasLog.csv"

# Read the CSV file
df = pd.read_csv(file_path)

pd_signal = np.array(df["CH2 - PK2Pk"])
norm_factor = np.array(df["CH1 - PK2Pk"])
norm_signal = pd_signal/norm_factor

delay_reflectivity = np.linspace(9.5,13.5,len(pd_signal))-10.5
delay_reflectivity = 2*delay_reflectivity/c
delay_reflectivity = np.around(delay_reflectivity, decimals=3)

maxw = find_index(delay_reflectivity,18)
shift = 10


plt.plot(delay_reflectivity[2:maxw],(norm_signal/np.max(norm_signal))[2+shift:maxw+shift],"go--",label="reflectivity",markersize=8,lw=1.5)
plt.show()

# %%

t = delay_reflectivity[2:maxw]
sig = (norm_signal/np.max(norm_signal))[2+shift:maxw+shift]

minw1 = find_index(sig,max(sig)) 
maxw1 = find_index(t, 10)

t = t[minw1:maxw1]
sig = sig[minw1:maxw1]

fit_sig, parameters = eft.fit_exp_decay(t, sig)
print(f"Tau: {parameters[1]*1e-12 :.2e} s")

plt.plot(delay_reflectivity[2:maxw],(norm_signal/np.max(norm_signal))[2+shift:maxw+shift],"go--",label="reflectivity",markersize=8,lw=1.5)
plt.plot(t,fit_sig,"r-",lw=5)
plt.xlabel("delay (ps)")
plt.ylabel("Normalized reflectivity")
plt.show()

