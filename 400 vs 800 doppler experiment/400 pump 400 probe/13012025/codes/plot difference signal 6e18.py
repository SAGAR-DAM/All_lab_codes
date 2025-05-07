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
mpl.rcParams['font.size'] = 25
mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=500 # highres display

# %%

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


# delays = np.array(sorted([30.4,29.65,31.15,31.9,32.65,33.4,28.9,30.7,31.45]))#,32.2,32.95,33.7,34.15,34.9,35.65]))
# delays = np.array(sorted([30.4,31.15,31.9,32.65,33.4,30.7]))#,32.2,32.95,33.7,34.15,34.9,35.65]))
delays = np.array(sorted([28.9,29.65,30.4,30.7,31.15,31.9,32.2,32.65,32.95,33.4,33.7,34.9,35.65]))
time_delays = (delays-30.4)*2/c



delay = "pr only"

files = glob.glob(f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\all delays\\{delay}\\*.txt")
f = open(files[0])
r=np.array(np.loadtxt(f,skiprows=17,comments='>'))

wavelength = r[:,0]
fullrange = len(wavelength)

w=np.array(wavelength)
I=np.zeros(fullrange)

for file in files[-4:]:
    f = open(file)
    r=np.array(np.loadtxt(f,skiprows=17,comments='>'))

    wavelength = r[:,0]
    intensity = r[:,1]


    intensity -= np.mean(intensity[0:200])
    intensity /= max(intensity)
    
    I += intensity/len(files[-4:])
    
minw = find_index(w, 407)
maxw = find_index(w, 410)
w_probe = w[minw:maxw]
I_probe = I[minw:maxw]

fit_I_probe,parameters,string = Gf.Gaussfit(w_probe, I_probe)

# %%

for i in range(9,10):
    # i = 0
    delay = delays[i]
    
    # files = glob.glob(f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\all delays\\{delay}\\*.txt")
    files = glob.glob(f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\values for plot\\{delay}\\*.txt")
    f = open(files[0])
    r=np.array(np.loadtxt(f,skiprows=17,comments='>'))
    
    wavelength = r[:,0]
    fullrange = len(wavelength)
    
    w=np.array(wavelength)
    I=np.zeros(fullrange)
    
    for file in files:
        f = open(file)
        r=np.array(np.loadtxt(f,skiprows=17,comments='>'))
    
        wavelength = r[:,0]
        intensity = r[:,1]
    
    
        intensity -= np.mean(intensity[0:200])
        intensity /= max(intensity)
        
        I += intensity/len(files)
        
    minw = find_index(w, 407)
    maxw = find_index(w, 410)
    w_delay = w[minw:maxw]
    I_delay = I[minw:maxw]
    fit_I_delay,parameters,string = Gf.Gaussfit(w_delay, I_delay)
    
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))  # 3 rows, 1 column
    
    # Plot 1
    axs[0].plot(w_probe,I_probe,"ro",markersize=7)
    # axs[0].plot(w_probe,fit_I_probe,"k-",lw=3)
    axs[0].set_title('Probe only spectrum',fontweight="bold")
    axs[0].text(x=407, y=0.5, s='(1)', color='black',fontweight="bold")
    axs[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axs[0].set_ylabel(r"$I_{norm}$")
    # Plot 2
    axs[1].plot(w_delay,I_delay,"bo",markersize=7)
    # axs[1].plot(w_delay,fit_I_delay,"k-",lw=3)
    axs[1].set_title(f'pump-probe spectrum',fontweight="bold")# \nat delay: {time_delays[i]:.0f} ps',fontweight="bold")
    axs[1].text(x=407, y=0.5, s='(2)', color='black',fontweight="bold")
    axs[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axs[1].set_ylabel(r"$I_{norm}$")
    
    # Plot 3
    axs[2].plot(w_probe,I_delay-I_probe,"ko",markersize=7)
    axs[2].set_title(f'difference signal (2)-(1)',fontweight="bold")
    axs[2].set_xlabel("wavelength (nm)")

    
    plt.tight_layout()
    plt.show()