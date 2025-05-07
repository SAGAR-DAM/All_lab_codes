# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:07:24 2023

@author: mrsag
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import numba

for i in range(36,37):
    f=open("D:\\data Lab\\frog\\Granouille allignment\\16jan2023\\Spectra\\5_%d.txt"%(i))  # read the data
    r=np.loadtxt(f)
    
    w=np.asarray(r[:,0])
    d=w[1]-w[0]
    
    
    f=(3*10**8*10**9)/w
    
    I=np.asarray(r[:,1])
    I1 = np.pad(I,(1024,1024),'constant',constant_values=(0,0))
    #I1 = I1 + 1000*np.random.random(len(I1))
    plt.figure(figsize = (37,20))
    plt.plot(w,I,linewidth=5,markersize=20)
    plt.title("Wavelength vs Intensity... position: %d" %i,size=75,fontname="cursive",fontweight="bold",color='green')
    plt.legend()
    plt.grid()
    plt.xticks(fontsize=45,color='purple')
    plt.yticks(fontsize=45,color='purple')
    plt.legend(fontsize=45)
    plt.xlabel("Wavelength",fontname="Times New Roman",fontweight="light",fontsize=60)
    plt.ylabel("Intensity",fontname="Times New Roman",fontweight="light",fontsize=60)
    plt.savefig('D:\\data Lab\\frog\\Granouille allignment\\16jan2023\\Spectra\\W vs I plots\\%d.jpg'%i)
    plt.show()
    
    
    plt.figure(figsize = (37,20))
    plt.plot(f,I,linewidth=5,markersize=20)
    plt.title("Frequency vs Intensity... position: %d" %i,size=75,fontname="cursive",fontweight="bold",color='green')
    plt.legend()
    plt.grid()
    plt.xticks(fontsize=45,color='purple')
    plt.yticks(fontsize=45,color='purple')
    plt.legend(fontsize=45)
    plt.xlabel("freq",fontname="Times New Roman",fontweight="light",fontsize=60)
    plt.ylabel("Intensity",fontname="Times New Roman",fontweight="light",fontsize=60)
    plt.savefig('D:\\data Lab\\frog\\Granouille allignment\\16jan2023\\Spectra\\W vs I plots\\freq %d.jpg'%i)
    plt.show()

    FFT = np.fft.fftshift(np.fft.fft(I,norm='ortho'))
    freq = np.fft.fftfreq(len(w),d=w[1]-w[0])
    plt.plot(((FFT.real)))
    #plt.xlim(1000,1040)
    plt.show()
    
    print(len(I))
    
    FFT_filter = FFT[950:1100]
    IFFT_filter = np.fft.ifft(FFT_filter.real,norm='ortho')
    plt.plot(IFFT_filter)