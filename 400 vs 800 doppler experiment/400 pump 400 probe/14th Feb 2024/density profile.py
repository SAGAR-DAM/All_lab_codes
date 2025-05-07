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
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=500 # highres display


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

######################################################################
# Simulation
######################################################################
c=0.3

# Load the dataset

files = glob.glob(r"D:\data Lab\400 vs 800 doppler experiment\Simulation with Jian\Hydro simulation 3e17 and 1e18 till 30 ps\TIFR_hydro_30ps\TIFR_1D_2_3e17_2\tifr_hdf5_plt_cnt_*")
step = 30
delay = np.linspace(0,30,len(files))

index = 160 # find_index(delay,16)
# delay = delay[:index]
files = files[:index]
pos = []

files = files[::step]
# delay = delay[::step]

i=0
for file in files:
    ds = yt.load(file)
    # Create a data object (like the entire domain)
    ad = ds.all_data()
    time = float(ds.current_time)*1e12
    index = find_index(np.array(ad['gas', 'El_number_density']),6.97e21)
    e_dens_in_10_e_21 = np.array(ad['gas', 'El_number_density'])#/1e21
    
    x = np.array(ad['gas', 'x'])*1e4
    
    pos.append(x[index])

    plt.plot(x,e_dens_in_10_e_21,label=f"{time:.1f} ps",lw=2)
    i+=1
    

plt.axhline(6.97e21,linestyle="--",color="k",lw=2)
plt.text(x=2.55,y=3e21,s="Critical Density line\n"+r"$n_c=6.97\times10^{21} cm^{-3}$")
plt.xlim(2,3)
plt.ylim(1e20,)
plt.yscale("log")
plt.legend()
plt.show()
    # %%


# pos[0] = 0.00025*1e4
# # pos = np.array(pos)-pos[0]
# t = np.linspace(0,30,len(pos))

# plt.plot(t,pos,"k-")
# plt.xlabel("delay (ps)")
# plt.ylabel("Simulation position of critical surface (um)")
# plt.show()

# dt = 1e-13
# vel_sim = -1e-4*np.diff(pos)/dt*1.556   # 1/cos(50) = 1.556 as the simulation was done on 50 degree AOI and velocity calculated along that angle
# plt.plot(t[1:],vel_sim)
# plt.xlabel("delay (ps)")
# plt.ylabel("Simulated velocity")
# plt.show()




# true_delay = t[1:][vel_sim!=0]
# true_vel = vel_sim[vel_sim!=0]

# x = moving_average(true_delay, 4)*0.95
# y = moving_average(true_vel, 4)

# y = y[0:find_index(x,18.5)]
# x = x[0:find_index(x,18.5)]


# # Threshold for spacing
# min_spacing = 1

# # Filtering logic
# filtered_x = [x[0]]
# filtered_y = [y[0]]
# last_x = x[0]

# for i in range(1, len(x)):
#     if abs(x[i] - last_x) >= min_spacing:
#         filtered_x.append(x[i])
#         filtered_y.append(y[i])
#         last_x = x[i]

