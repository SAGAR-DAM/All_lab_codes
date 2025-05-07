# -*- coding: utf-8 -*-
"""
Created on Sat May  3 18:38:05 2025

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
from Curve_fitting_with_scipy import exp_decay_fit as eft
import pandas as pd

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=500 # highres display


def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index


files = glob.glob(r"D:\data Lab\400 vs 800 doppler experiment\Simulation with Jian\Result for hydro simulation 09-04-2025\TIFR_hydro\TIFR_hydro\TIFR_1D_2_3e17\tifr_hdf5_plt_cnt_*")
files = files[1:100]

Te_arr = []
pos = []

for i in range(4,len(files)):
    file = files[i]
    ds = yt.load(file)

    ad = ds.all_data()

    ne = np.array(ad['gas', 'El_number_density'])
    maxw = find_index(ne,6.97e21)
    ne = ne[:maxw]
    
    # ni = np.array(ad['gas', 'ion_number_density'])
    # ni = ni[:maxw]
    
    Te = np.array(ad["flash","tele"])/11604   # code temperature in Kelvin. devide by 11604 to get eV
    Te = Te[:maxw]
    
    x = (np.array(ad['flash', 'x'])[:maxw])
    pos.append(x[-1])
    
    weight = ne/np.sum(ne)
    weighted_Te = Te*weight
    
    Te_arr.append(np.sum(weighted_Te))
    
    # plt.plot(Te,label="Te")
    # plt.plot(weight*max(Te)/max(weight),label="weight from ion density")
    # plt.legend()
    # plt.show()

# %%

t = np.linspace(0,10.03,len(Te_arr))

plt.plot(t,Te_arr,"r-",lw=5,label=r"$\overline{Te(t)}$")
plt.title("average electron temperature\n over plasma length")
plt.xlabel("delay (ps)")
plt.text(x=5,y=80,s=r"$\langle\overline{Te(t)}\rangle_t$: "+f"{np.mean(Te_arr):.2f} ev")
plt.ylabel("Te (eV)")
plt.legend()
plt.show()

# %%

# weight = pos/np.sum(pos)
# avg_Z = np.sum(weight*Z_arr)
# print(avg_Z)