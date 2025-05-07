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
mpl.rcParams['font.size'] = 20
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display



f = open(r"D:\data Lab\400 vs 800 doppler experiment\contrast.txt")
r=np.loadtxt(f)

# If data has two columns: x and y
x = r[:, 0]
y = r[:, 1]

x = x[::8]*1e12
y = y[::8]
y /= max(y)

# Plot
plt.figure(figsize=(6,4))
plt.plot(x,y,"r-",label="800 nm",lw=2)
plt.plot(x,y**2,"b-",label="400 nm",lw=2)
plt.xlabel('time (ps)')
plt.ylabel('Intensity level (arb unit)')
plt.yscale("log")
plt.ylim(min(y**2),1)
plt.xlim(-400,)
plt.minorticks_on()
# plt.grid(lw=1,color="k")
# plt.grid(which="minor",lw=0.5,color="k")
plt.title("Intensity contrast\n800 nm: Measured, 400 nm: theoretical\n")
plt.legend()
plt.show()
