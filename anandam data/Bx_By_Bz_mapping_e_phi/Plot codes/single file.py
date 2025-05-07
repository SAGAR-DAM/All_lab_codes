# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:49:58 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt


import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

t = '0.33'  
Bz = '80.0'
filename = f"D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\t_{t}\\ellipticity\\Bx_By_to_ellipticity_t_{t}_Bz_{Bz}.txt"

f = open(filename,"r")
r = np.loadtxt(f)
r = np.flip(r,axis=0)

plt.imshow(r, cmap="jet", extent = [-100,100,-100,100])
plt.title(r"ellipticity (|tan $\chi$|) vs $\vec B_{trans}$"+"\n"+f"Bz={Bz} MG; "+r"L$_{plasma}$="+f"{float(t)*1e-12*6e6*1e7 :.2f} nm")
plt.colorbar()
plt.xlabel("Bx (MG)")
plt.ylabel("By (MG)")
plt.show()

filename = f"D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\t_{t}\\tan_psi\\Bx_By_to_tan_faraday_rot_t_{t}_Bz_{Bz}.txt"

f = open(filename,"r")
r = np.loadtxt(f)
r = np.flip(r,axis=0)

plt.imshow(r, cmap="jet", extent = [-100,100,-100,100])
plt.title(r"faraday rotation (|tan $\psi$|) vs $\vec B$"+"\n"+f"Bz={Bz} MG; "+r"L$_{plasma}$="+f"{float(t)*1e-12*6e6*1e7 :.2f} nm")
plt.colorbar()
plt.xlabel("Bx (MG)")
plt.ylabel("By (MG)")
plt.show()

