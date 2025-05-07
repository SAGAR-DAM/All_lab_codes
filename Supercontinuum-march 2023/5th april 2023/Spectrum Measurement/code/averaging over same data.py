# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 09:37:23 2023

@author: sagar
"""

'''
Spectrum with averaging over several sets.
'''

import numpy as np
import matplotlib.pyplot as plt
import glob

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display


def main():
    files=glob.glob("D:\\data Lab\\Supercontinuum-march 2023\\5th april 2023\\Spectrum Measurement\\data\\*.trt")
    fileset=[4,5,5,5]      #number of files for which the  averaging is carried out
    
    rangearr=[450,1400]           #range of the x axis data needed
    fileindex=0                  
    
    intensity_matrix=[]          #store the arrays to plot all the plots together
    labels=[]                    # labels for different plots at together plot
    
    for i in range(len(fileset)):
        wavelength=np.zeros(rangearr[1]-rangearr[0])
        intensity=np.zeros(rangearr[1]-rangearr[0])
        
        string=""
        for k in range(fileset[i]):
            f=open(files[fileindex],'r')
            r=np.loadtxt(f,skiprows=8,delimiter=';')
            if(k==0):
                wavelength=r[:,0][rangearr[0]:rangearr[1]]
                print(min(wavelength),wavelength[1]-wavelength[0])
                filename=files[fileindex]
                file=open(filename,'r')
                heading=file.readlines()
                h=heading[0]
                filterinfo=heading[1]
                filters=[]
                
                for j in range(int(len(filterinfo)/2)):
                    y=float(filterinfo[2*j:2*j+2])
                    if(y==0):
                        filters.append(100)
                    else:
                        filters.append(y)
                filters=np.array(filters)
                filters=filters/100
                
                factor=1
                for j in range(len(filters)):
                    factor=factor*filters[j]
                    
                    
            intensity=intensity+np.array(r[:,1][rangearr[0]:rangearr[1]])
            string=string+"+"+files[fileindex][-11:]
            
            fileindex+=1
            
            
        intensity=intensity/(fileset[i]*factor)
        
        intensity_matrix.append(intensity)
        labels.append(str(h[2:len(h)-1]))
        
        plt.figure()
        plt.plot(wavelength,intensity)
        plt.title(f"{string[1:]}",fontsize=7)
        plt.suptitle("Supercontinuum Spectrum of files:")
        plt.grid()
        plt.xlabel(f"Wavelength (nm)\n{h[2:len(h)-1]}", fontname='Times New Roman')
        plt.ylabel("Intensity (Arb unit)", fontname='Times New Roman')
        plt.savefig("D:\\data Lab\\Supercontinuum-march 2023\\5th april 2023\\Spectrum Measurement\\averaged over plots\\%s.jpg"%(h[2:len(h)-1]),bbox_inches='tight')
        plt.show()
        
    plt.figure()
    for i in range(len(labels)):
        if i==0:
            minval=525
            maxval=720
        elif i==1:
            minval=374
            maxval=525
        elif i==2:
            minval=0
            maxval=374
        elif i==3:
            minval=720
            maxval=len(wavelength)-1
        
        plt.plot(wavelength[minval:maxval],intensity_matrix[i][minval:maxval],label=labels[i])
        plt.grid()
        plt.xlabel(f"Wavelength (nm)", fontname='Times New Roman')
        plt.ylabel("Intensity (Arb unit)", fontname='Times New Roman')
        plt.yscale('log')
        plt.title("Comparisn of different supercontinuums (Un-Normalized)")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.savefig("D:\\data Lab\\Supercontinuum-march 2023\\5th april 2023\\Spectrum Measurement\\averaged over plots\\%s.jpg"%('Comparisn of different supercontinuums Un-Normalized'),bbox_inches='tight')
    plt.show()
    
    plt.figure()
    for i in range(len(labels)):
        
        plt.plot(wavelength,intensity_matrix[i]/max(intensity_matrix[i]),label=labels[i])
        plt.grid()
        plt.xlabel(f"Wavelength (nm)", fontname='Times New Roman')
        plt.ylabel("Intensity (Arb unit)", fontname='Times New Roman')
        plt.title("Comparisn of different supercontinuums (Normalized)")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.savefig("D:\\data Lab\\Supercontinuum-march 2023\\5th april 2023\\Spectrum Measurement\\averaged over plots\\%s.jpg"%('Comparisn of different supercontinuums Normalized'),bbox_inches='tight')
    plt.show()
        
if __name__=='__main__':
    main()