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


filename = "D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\t_5.33\\ellipticity\\Bx_By_to_ellipticity_t_5.33_Bz_41.0.txt"

f = open(filename,"r")
r = np.loadtxt(f)
r = np.flip(r,axis=0)

plt.imshow(r, cmap="jet", extent = [-100,100,-100,100])
plt.title(r"ellipticity vs $\vec B$")
plt.colorbar()
plt.show()

