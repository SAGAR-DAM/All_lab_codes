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

def main():
    files=glob.glob("D:\\data Lab\\Supercontinuum-march 2023\\14th march SCG\\Spectrum Measurement\\data\\*.trt")
    for i in range(len(files)):
        f=open(files[i],'r')
        r=np.loadtxt(f,skiprows=8,delimiter=';')
        rangearr=[450,900]
        wavelength=r[:,0][rangearr[0]:rangearr[1]]
        intensity=r[:,1][rangearr[0]:rangearr[1]]
        
        filename=files[i]
        file=open(filename,'r')
        heading=file.readlines()
        h=heading[0]
        filterinfo=heading[1]
        filters=[]
        print(h)
        for j in range(int(len(filterinfo)/2)):
            y=int(filterinfo[2*j:2*j+2])
            if(y==0):
                filters.append(100)
            else:
                filters.append(y)
        filters=np.array(filters)
        filters=filters/100
        
        factor=1
        for j in range(len(filters)):
            factor=factor*filters[j]
        
        intensity=intensity/factor
        
        plt.figure()
        plt.plot(wavelength,intensity)
        plt.title(f"Spectrum of {files[i][-11:]}    Filters: {100*filters}")
        plt.suptitle("%s"%h)
        plt.grid()
        plt.xlabel("Wavelength (nm)", fontname='Times New Roman')
        plt.ylabel("Intensity (Arb unit)", fontname='Times New Roman')
        plt.savefig("D:\\data Lab\\Supercontinuum-march 2023\\14th march SCG\\Spectrum Measurement\\plots\\%s.jpg"%(files[i][-11:]),bbox_inches='tight')
        plt.show()
        
        #print(heading)
        
if __name__=="__main__":
    main()