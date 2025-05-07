# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:50:11 2023

@author: sagar

"""
import glob
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display


files=glob.glob("D:\\data Lab\\Supercontinuum-march 2023\\13th march 2023\\Spectrum Measurement\\13th_March_SCG\\spectrum\\*.trt")
for i in range(len(files)):
    f=open(files[i],'r')
    r=np.loadtxt(f,skiprows=7,delimiter=';')
    wavelength=r[:,0]
    intensity=r[:,1]
    
    filename=files[i]
    file=open(filename,'r')
    heading=file.readlines()
    h=heading[0]
    print(h)
    
    
    plt.figure()
    plt.plot(wavelength,intensity)
    plt.title(f"Spectrum of {files[i][-15:]}")
    plt.suptitle("%s"%files[i][-15:]+"    =     %s"%h)
    plt.grid()
    plt.xlabel("Wavelength (nm)", fontname='Times New Roman')
    plt.ylabel("Intensity (Arb unit)", fontname='Times New Roman')
    plt.savefig("D:\\data Lab\\Supercontinuum-march 2023\\13th march 2023\\Spectrum Measurement\\plots\\%s.jpg"%(files[i][-15:]),bbox_inches='tight')
    plt.show()
    
    #print(heading)