# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 09:37:23 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display


def main():
    files=glob.glob("D:\\data Lab\\Supercontinuum-march 2023\\13th march 2023\\Spectrum Measurement\\13th_March_SCG\\spectrum\\*.trt")
    
    for i in range(len(files)):
        f=open(files[i],'r')
        r=np.loadtxt(f,dtype=np.str,skiprows=7)
        
        filename=files[i]
        file=open(filename,'r')
        heading=file.readlines()
        h=heading[0]
        print(h)
        
        data=r[0:]
        wavelength=[]
        intensity=[]
        
        for j in range(len(data)):
            semicolon_index=(data[j]).index(';')
            
            d=float(data[j][:semicolon_index-1])
            wavelength.append(d)
            
            d=float(data[j][semicolon_index+1:])
            intensity.append(d)
            
        
        #print(data)
        #print(float(data[0][7:]))
        
        plt.figure()
        plt.plot(wavelength,intensity)
        plt.title(f"Spectrum of {files[i][-15:]}")
        plt.suptitle("%s"%files[i][-15:]+"    =     %s"%h)
        plt.grid()
        plt.xlabel("Wavelength (nm)", fontname='Times New Roman')
        plt.ylabel("Intensity (Arb unit)", fontname='Times New Roman')
        plt.savefig("D:\\data Lab\\Supercontinuum-march 2023\\13th march 2023\\Spectrum Measurement\\plots\\%s.jpg"%(files[i][-15:]),bbox_inches='tight')
        plt.show()


if __name__=='__main__':
    main()