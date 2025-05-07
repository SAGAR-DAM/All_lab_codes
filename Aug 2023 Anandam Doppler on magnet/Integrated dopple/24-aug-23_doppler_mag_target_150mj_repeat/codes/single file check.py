# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 13:39:27 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
from decimal import Decimal
import glob

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

def nearest_index(arr,x):
    index=(abs(np.array(arr)-x)).argmin()
    return(index)

def Gauss1(x,b,x0):
    y=np.exp(-(x-x0)**2/(2*b**2))
    return y

def Gaussfit(w,I):
    xdata=w         #Taking the x axis data
    ydata=I         #Taking the y axis data
    
    ''' 
        here the code fits only the normalized Gaussian
        So, we first normalize the array and later multiply with the amplitude factor to get the main array
    '''
    y_maxval=max(ydata)      #taking the maximum value of the y array
    ymax_index=(list(ydata)).index(y_maxval)   
    
    xmax_val=xdata[ymax_index]  #Shifting the array as a non-shifted Gausian 
    xdata=xdata-xmax_val        #Shifting the array as a non-shifted Gausian
    
    ydata=ydata/y_maxval
    
    parameters, covariance = fit(Gauss1, xdata, ydata,maxfev=100000)
    fit_y = Gauss1(xdata, *parameters)
    
    
    xdata=xdata+xmax_val
    parameters[1]+=xmax_val
    
    fit_y=np.asarray(fit_y)
    fit_y=fit_y*y_maxval       # again multiplying the data to get the actual value
    
    string1=r"Fit: $f(x)=Ae^{-\frac{(x-x_0)^2}{2b^2}}$;"
    string2=rf"with A={Decimal(str(y_maxval)).quantize(Decimal('1.00'))}, b={Decimal(str(parameters[0])).quantize(Decimal('1.00'))}, $x_0$={Decimal(str(parameters[1])).quantize(Decimal('1.00'))}"
    string=string1+string2
    return fit_y,parameters,string

filename="D:\\data Lab\\Aug 2023 Anandam Doppler on magnet\\Integrated dopple\\24-aug-23_doppler_mag_target_150mj_repeat\\good files\\magh_rep010.txt"

f=open(filename,'r')
r=np.loadtxt(f,skiprows=17,comments=">")

intensity_base=np.mean(r[:,1][0:1000])

wavelength=r[:,0]
minw=nearest_index(wavelength, 400)
maxw=nearest_index(wavelength, 416)
wavelength=wavelength[minw:maxw]

intensity=r[:,1][minw:maxw]-intensity_base

fit_y,parameters,info_string=Gaussfit(wavelength,intensity)

print(parameters)

plt.plot(wavelength,intensity)
plt.plot(wavelength,fit_y)
plt.title(r"%s"%info_string)
plt.show()