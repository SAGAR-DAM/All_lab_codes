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
    files=glob.glob("D:\\data Lab\\Supercontinuum-march 2023\\comparisn of 5th and 6th april fs vs GaO\\Spectrum Measurement\\data\\*.trt")
    fileset=[4,5,5,5,5,5,5,5,5]      #number of files for which the  averaging is carried out
    
    rangearr=[300,1400]           #range of the x axis data needed
    fileindex=0                  
    
    intensity_matrix=[]          #store the arrays to plot all the plots together
    labels=[]                    # labels for different plots at together plot
    
    for i in range(len(fileset)):
        if(i<=3):
            rangearr=[450,1400]
        else:
            rangearr=[300,1400]
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
                    
                if(i==2):
                    factor=factor*25
                if(i==3):
                    factor=factor*5
                    
                    
            intensity=intensity+np.array(r[:,1][rangearr[0]:rangearr[1]])
            string=string+"+"+files[fileindex][-11:]
            
            fileindex+=1
            
            
        intensity=intensity/(fileset[i]*factor)
        
        intensity_matrix.append(intensity)
        labels.append(str(h[2:len(h)-1]))
        

    plt.figure()
    for i in range(len(labels)):
        if i==0:
            minval=525
            maxval=720
        elif i==1:
            minval=360
            maxval=525
        elif i==2:
            minval=0
            maxval=360
        elif i==3:
            minval=720
            maxval=949-1
        elif i==4:
            minval=658
            maxval=880
        elif i==5:
            minval=50
            maxval=492
        elif i==6:
            minval=575
            maxval=658
        elif i==7:
            minval=880
            maxval=len(wavelength)-1
        elif i==8:
            minval=492
            maxval=575
        
        if i<=3:
            if(i==3):
                plt.plot(wavelength[minval:maxval]+88,intensity_matrix[i][minval:maxval],'k-',label="Fused Silica/ energy: 52 $\mu J$")
            plt.plot(wavelength[minval:maxval]+88,intensity_matrix[i][minval:maxval],'k-')
        else:
            plt.plot(wavelength[minval:maxval],intensity_matrix[i][minval:maxval],'r-')
            if(i==5):
                plt.plot(wavelength[minval:maxval],intensity_matrix[i][minval:maxval],'r-',label=r"$Ga_2O_3$"+"/ energy: 52 $\mu J$")
        #plt.plot(wavelength[minval:maxval],intensity_matrix[i][minval:maxval],label=labels[i])
        plt.grid()
        plt.xlabel(f"Wavelength (nm)", fontname='Times New Roman')
        plt.ylabel("Intensity (Arb unit)", fontname='Times New Roman')
        plt.yscale('log')
        plt.title("Comparisn of different supercontinuums (Un-Normalized)")
        plt.legend()
            
    plt.savefig("D:\\data Lab\\Supercontinuum-march 2023\\comparisn of 5th and 6th april fs vs GaO\\Spectrum Measurement\\averaged over plots\\%s.jpg"%('Comparisn of different supercontinuums Un-Normalized'),bbox_inches='tight')
    plt.show()
    
    plt.figure()
    
if __name__=='__main__':
    main()