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

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.weight'] = 'bold'
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
    filtered_signal = fftconvolve(signal, window, mode='valid')
    
    return filtered_signal

# %%
min_wavelength = 408
max_wavelength = 413
c = 0.3
files_17 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_17\\scan 1\\*.txt"))
files_17 = files_17[::2]
peaks1 = []
delay1 = np.linspace(12,15,len(files_17))-12.65
delay1 = 2*delay1/c
delay1 = np.around(delay1, decimals=3)

# %%
i=0
f = open(files_17[i])
r=np.loadtxt(f,skiprows=17,comments='>')

wavelength = r[:,0]
intensity = r[:,1]
intensity /= max(intensity)
intensity -= np.mean(intensity[0:200])

minw = find_index(wavelength, min_wavelength)
maxw = find_index(wavelength, max_wavelength)

wavelength = wavelength[minw:maxw]
intensity = intensity[minw:maxw] 


intensity /= max(intensity)

fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)

w_probe = wavelength
I_probe = intensity

fit_I_probe,parameters_probe,string = Gf.Gaussfit(w_probe, I_probe)

# %%

i=find_index(delay1,12)
f = open(files_17[i])
r=np.loadtxt(f,skiprows=17,comments='>')

wavelength = r[:,0]
intensity = r[:,1]
intensity /= max(intensity)
intensity -= np.mean(intensity[0:200])

minw = find_index(wavelength, min_wavelength)
maxw = find_index(wavelength, max_wavelength)

wavelength = wavelength[minw:maxw]
intensity = intensity[minw:maxw] 


intensity /= max(intensity)

fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)

w_delay = wavelength
I_delay = intensity

fit_I_delay,parameters_delay,string = Gf.Gaussfit(w_delay, I_delay)

# %%

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))  # 3 rows, 1 column
axs[0].plot(w_probe,I_probe,"o",color="g",markersize=7,label="Signal")
axs[0].plot(w_probe,fit_I_probe,"k--",lw=3,label="Fit")
axs[0].set_title('Probe only spectrum',fontweight="bold")
axs[0].text(x=408, y=0.5, s='(1)', color='black',fontweight="bold")
axs[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
axs[0].set_ylabel(r"$I_{norm}$")
axs[0].legend()
# Plot 2
axs[1].plot(w_delay,I_delay,"r",marker="^",linestyle='None',markersize=7,label="signal")
axs[1].plot(w_delay,fit_I_delay,"k--",lw=3,label="Fit")
axs[1].set_title(f'Pump-probe spectrum',fontweight="bold")# \nat delay: {time_delays[i]:.0f} ps',fontweight="bold")
axs[1].text(x=408, y=0.5, s='(2)', color='black',fontweight="bold")
axs[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
axs[1].set_ylabel(r"$I_{norm}$")
axs[1].legend(loc="upper right")
# Plot 3
axs[2].plot(w_probe,I_delay-I_probe,"ko",markersize=7)
axs[2].set_title(f'Difference signal: (2) - (1)',fontweight="bold")
axs[2].set_xlabel("wavelength (nm)")


plt.tight_layout()
plt.show()

# %%
avg_w = moving_average(w_probe,3)
difference_sig = moving_average(I_delay-I_probe,3)

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))  # 3 rows, 1 column
axs[0].plot(w_probe,I_probe,"o",color="g",markersize=10,label="Signal")
axs[0].plot(w_probe,fit_I_probe,"k--",lw=3,label="Fit")
axs[0].axvline(x=parameters_probe[1],color="g",linestyle="--",linewidth=2)
axs[0].text(x=408,y=0.8,s='Probe only spectrum',fontweight="bold")
axs[0].text(x=408, y=0.5, s='(1)', color='black',fontweight="bold")
axs[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
axs[0].set_ylabel(r"$I_{norm}$")
axs[0].legend()
# Plot 2
axs[1].plot(w_delay,I_delay,"r",marker="^",linestyle='None',markersize=10,label="Signal")
axs[1].plot(w_delay,fit_I_delay,"k--",lw=3,label="Fit")
axs[1].axvline(x=parameters_delay[1],color="r",linestyle="--",linewidth=2)
axs[1].text(x=408,y=0.8,s=f'Pump-probe spectrum',fontweight="bold")# \nat delay: {time_delays[i]:.0f} ps',fontweight="bold")
axs[1].text(x=408, y=0.5, s='(2)', color='black',fontweight="bold")
axs[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
axs[1].set_ylabel(r"$I_{norm}$")
axs[1].legend(loc="upper right")
# Plot 3
axs[2].plot(avg_w,difference_sig,"ko",markersize=10)
axs[2].axhline(y=0,color="b",linestyle="--",linewidth=2)
axs[2].axvline(x=410.7,color="b",linestyle="--",linewidth=2)
axs[2].text(x=408,y=0.4,s=f'Difference signal: (2) - (1)',fontweight="bold")
axs[2].text(x=408, y=0.25, s='(3)', color='black',fontweight="bold")
axs[2].set_xlabel("wavelength (nm)")


plt.tight_layout()
plt.show()
