# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:24:13 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from Curve_fitting_with_scipy import Gaussianfitting as Gf

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

c = 0.3   #in mm/ps

files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14 th march 2024\\Scan3\\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run9_70%_20TW_ret_11-15_250fs\\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run8_30%_20TW_ret_11-15_250fs\\*.txt")

delay = np.linspace(10,13.432,len(files)//2)-11.35
delay = 2*delay/c
delay = np.around(delay, decimals=3)

peaks = []

def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index


# for i in range(1,len(files)-2,2):
#     f = open(files[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
#     peaks.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i]}")
#     plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

for i in range(1,len(files)-2,2):
    if(i<=161):
        f = open(files[i])
    else:
        f = open(files[i-1])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 390)
    maxw = find_index(wavelength, 420)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i//2]}")
    # plt.xlim(410,416)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.grid(lw = 1, color = "black")
    # plt.show()    
    
plt.plot(delay[0:len(peaks)],peaks, 'ro-')
#plt.xlim(-6,max(delay))
plt.ylim(412.7,413.8)
plt.xlabel("delay (ps)")
plt.ylabel("peak wavelength (nm)")
plt.grid(lw = 1, color = "black")
plt.title("Peak Wavelength")
plt.show()

