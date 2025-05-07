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
from skimage import io 

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 32
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=500 # highres display

f1 = open(r"D:\data Lab\400 vs 800 doppler experiment\400 nm focal spot\23122024_400nm\horizontal cut.txt")
r=np.loadtxt(f1,skiprows=17,comments='>')
h_data = r[:,1]

f2 = open(r"D:\data Lab\400 vs 800 doppler experiment\400 nm focal spot\23122024_400nm\vertical cut.txt")
r=np.loadtxt(f2,skiprows=17,comments='>')
v_data = r[:,1]

range1 = np.arange(len(h_data))+16
range2 = np.arange(len(v_data))+16


image = io.imread(r"D:\data Lab\400 vs 800 doppler experiment\400 nm focal spot\23122024_400nm\image3.png")
image = np.array(image)

plt.figure(figsize=(6,6))
plt.imshow(image,origin="upper",cmap="plasma")
plt.plot(range1,25+h_data/2,color="w")
plt.plot(25+v_data/2,range2,color="y")
plt.axis("off")
plt.colorbar()
plt.show()
