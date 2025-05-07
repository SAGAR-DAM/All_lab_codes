# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:01:57 2025

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




delays = np.array(sorted([28.9,29.65,30.4,30.7,31.15,31.9,32.2,32.65,32.95,33.4,33.7,34.9,35.65]))
time_delays = (delays-30.4)*2/c
peaks = []
std = []

for delay in delays:
    files = glob.glob(f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\values for plot\\{delay}\\*.txt")
    peak = []
    
    for file in files:
        f = open(file)
        r=np.array(np.loadtxt(f,skiprows=17,comments='>'))
        
        wavelength = r[:,0]
        intensity = r[:,1]
        
        
        intensity -= np.mean(intensity[0:200])
        intensity /= max(intensity)
        minw = find_index(wavelength, 400)
        
        maxw = find_index(wavelength, 420)
        
        wavelength = wavelength[minw:maxw]
        intensity = intensity[minw:maxw] 
        
        fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
        
        peak.append(np.mean(wavelength[intensity>0.95]))
        
    peaks.append(np.mean(peak))
    std.append(max([np.std(peak)*2,0.03]))
    
    
peaks = np.array(peaks)
std = np.array(std)/2
shifts = peaks-np.mean(peaks[0:3])

blue_delay = time_delays[shifts<0]
blue_shifts = shifts[shifts<0]

red_delay = time_delays[shifts>0]
red_shifts = shifts[shifts>0] 

plt.errorbar(time_delays, shifts, yerr=std,lw=0, elinewidth=2, capsize=2, color = 'r')
plt.errorbar(time_delays, shifts, yerr=std,lw=0, elinewidth=1, capsize=2, color = 'k')
plt.plot(time_delays, shifts,'ko-')
plt.plot(blue_delay,blue_shifts,"bo")
plt.plot(red_delay, red_shifts,"ro")
plt.plot(time_delays, shifts,'ko', markersize=2)
plt.grid(lw=0.3,color='k')
plt.xlabel("delay (ps)")
plt.ylabel("spectral shift (nm)")
plt.title("Pu-Pr delay vs probe doppler shift")


# Generate sample data
x = np.linspace(min(time_delays)*1.1, max(time_delays)*1.1, 100)
y1 = np.ones(len(x))*(max(shifts+std))*1.1
y2 = np.ones(len(x))*(min(shifts-std))*1.1

plt.xlim(min(time_delays)*1.1, max(time_delays)*1.05)
plt.ylim((min(shifts-std))*1.1,(max(shifts+std))*1.1)

# Set background colors based on y-values
plt.fill_between(x, y2, where=(y2 <= 0), color='b', alpha=0.2)
plt.fill_between(x, y1, where=(y1 > 0), color='r', alpha=0.2)

plt.show()











def calc_vel(w, w0):
    c_norm = 3e10
    v = (w**2-w0**2)/(w**2+w0**2)*c_norm
    return v

lambda0 = np.mean(peaks[0:3])


velocity = np.array(calc_vel(np.array(peaks), lambda0))
velocity_err = np.abs(np.array(calc_vel(peaks+std, lambda0))-velocity)

blue_v = velocity[shifts<0]
red_v = velocity[shifts>0]


plt.errorbar(time_delays,velocity, yerr=velocity_err,lw=0, elinewidth=2, capsize=2, color = 'r')
plt.errorbar(time_delays, velocity, yerr=velocity_err,lw=0, elinewidth=1, capsize=2, color = 'k')
plt.plot(time_delays,velocity,'ko-')
plt.plot(blue_delay,blue_v,"bo")
plt.plot(red_delay, red_v,"ro")
plt.plot(time_delays,velocity,'ko', markersize=2)
plt.grid(lw=0.3,color='k')
plt.xlabel("delay (ps)")
plt.ylabel(r"v (cm/s)")
plt.title("Pu-Pr delay vs dopler velocity at: "+r"$I=6\times 10^{18}\ W/cm^2$"+"\n(red means inside the target velocity)")


# Generate sample data
x = np.linspace(min(time_delays)*1.1, max(time_delays)*1.1, 100)
y1 = np.ones(len(x))*(max(velocity+velocity_err))*1.1
y2 = np.ones(len(x))*(min(velocity-velocity_err))*1.1

plt.xlim(min(time_delays)*1.1, max(time_delays)*1.05)
plt.ylim((min(velocity-velocity_err))*1.1,(max(velocity+velocity_err))*1.1)

# Set background colors based on y-values
plt.fill_between(x, y2, where=(y2 <= 0), color='b', alpha=0.2)
plt.fill_between(x, y1, where=(y1 > 0), color='r', alpha=0.2)

plt.show()






data1 = np.column_stack((time_delays,velocity,velocity_err))

# Open a file to write
with open("D:\\output.txt", "w") as f:
    # Write the first section
    f.write("""Intensity: 6e18 W/cm^2

Scan 1
delay | velocity | error_v\n""")
    np.savetxt(f, data1, fmt="%.2f", delimiter=", ")
    
    # Add a blank line or separator
    f.write("\n\n")