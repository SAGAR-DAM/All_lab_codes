# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 13:39:27 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
import scipy.stats as stats
from decimal import Decimal
import glob

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display



#########################################################################
#########################################################################
''' Defining required functions '''
#########################################################################
#########################################################################

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

def chi_sq(expected, observed):
    return stats.chisquare(expected, np.sum(expected)/np.sum(observed) * observed)

def relative_error(x,y):
    x=np.array(x)
    y=np.array(y)
    x_y=x-y
    
    error_arr=x_y**2
    error_arr=error_arr/len(x)
    abs_error=np.sqrt(np.sum(error_arr))
    
    return(abs_error)

# Define the parabolic function
def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def parabolic_fit(x, y):
    # Fit the data to the parabolic function
    params, covariance = fit(parabola, x, y)

    # Extracting the coefficients
    a, b, c = params

    return a, b, c

#########################################################################
#########################################################################
''' Input of required files to get spectrum'''
#########################################################################
#########################################################################

files=glob.glob("D:\\data Lab\\Single shot SD\\11_oct_2023_SD_Frog\\400nm before crystal\\*.txt")

peak_wavelength=[]

for i in range(len(files)):
    f=open(files[i],'r')
    r=np.loadtxt(f,skiprows=17,comments=">")
    
    minw=700
    maxw=1300
    #print(maxw)
    
    intensity_base=np.mean(r[:,1][0:100])
    
    wavelength=r[:,0][minw:maxw]
    intensity=r[:,1][minw:maxw]-intensity_base
    intensity=intensity/max(intensity)
    
    fit_y,parameters,info_string=Gaussfit(wavelength,intensity)
    peak_wavelength.append(parameters[1])
    
laser_wavelength=np.mean(peak_wavelength)
print(laser_wavelength)



for __var__ in dir():
    exec('del '+ __var__)
    del __var__