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



#########################################################################
#########################################################################
''' Defining required functions '''
#########################################################################
#########################################################################

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

#########################################################################
#########################################################################
''' Input of required files to get spectrum'''
#########################################################################
#########################################################################

files=glob.glob("D:\\data Lab\\Aug 2023 Anandam Doppler on magnet\\Integrated dopple\\24-aug-23_doppler_mag_target_150mj_repeat\\good files\\*.txt")

set_numbers=[5,3,3,3,3,3,3,2,2,3,3,4,3,1]
retro_pos=np.array([0,6.2,6.35,6.5,6.65,6.8,6.95,7.1,7.25,7.4,7.55,7.7,7.85,8])-7.1

peak_wavelength=[]
blue=[]
blue_index=[]
red=[]
red_index=[]

#########################################################################
#########################################################################
''' scanning through all files and taking maximum'''
#########################################################################
#########################################################################

file_index=0

for i in range(len(set_numbers)):
    pw=[]
    
    for j in range(set_numbers[i]):
        
        f=open(files[file_index],'r')
        r=np.loadtxt(f,skiprows=17,comments=">")
        
        filename=files[file_index][-15:]
        
        intensity_base=np.mean(r[:,1][0:1000])
        
        wavelength=r[:,0]
        minw=nearest_index(wavelength, 404)
        maxw=nearest_index(wavelength, 412)
        wavelength=wavelength[minw:maxw]
        
        intensity=r[:,1][minw:maxw]-intensity_base
        
        fit_y,parameters,info_string=Gaussfit(wavelength,intensity)
        
        print(parameters)
        pw.append(parameters[1])
        
        plt.plot(wavelength,intensity)
        plt.plot(wavelength,fit_y)
        plt.title(filename+"\n"+r"%s"%info_string)
        plt.show()
        
        file_index+=1
        
    peak_wavelength.append(np.mean(pw))

for i in range(1,len(peak_wavelength)):
    if(peak_wavelength[i]>peak_wavelength[0]):
        red_index.append(retro_pos[i])
        red.append(peak_wavelength[i])
    else:
        blue.append(peak_wavelength[i])
        blue_index.append(retro_pos[i])

#########################################################################
#########################################################################
''' plotting all doppler shift'''
#########################################################################
#########################################################################

level=np.ones(len(peak_wavelength))*peak_wavelength[0]
level_index=np.linspace(np.min(retro_pos[1:]),np.max(retro_pos[1:]),len(level))

plt.plot(retro_pos[1:],peak_wavelength[1:],'ko-')
plt.plot(blue_index,blue,'bo')
plt.plot(red_index,red,'ro')
plt.plot(level_index,level,'k--')
plt.title("Doppler shift vs retro position")
plt.xlabel("Retro postion (mm) \n(w.r.t time zero)")
plt.ylabel("Peak Wavelength (nm)")
plt.grid(True)
plt.show()


#########################################################################
#########################################################################
''' plotting all doppler shift w.r.t zero (absolute shift)'''
#########################################################################
#########################################################################

lambda_0=peak_wavelength[0]
delta_wavelength=np.array(peak_wavelength)-lambda_0
red=np.array(red)-lambda_0
blue=np.array(blue)-lambda_0

c=3*10**(-1) # in mm/ps
delay=np.array(retro_pos)*2/c
red_delay=np.array(red_index)*2/c
blue_delay=np.array(blue_index)*2/c

level=np.ones(len(delta_wavelength))*delta_wavelength[0]
level_index=np.linspace(np.min(delay[1:]),np.max(delay[1:]),len(level))

plt.plot(delay[1:],delta_wavelength[1:],'ko-')
plt.plot(blue_delay,blue,'bo')
plt.plot(red_delay,red,'ro')
plt.plot(level_index,level,'k--')
plt.title("Doppler shift vs delay")
plt.xlabel("delay (ps)")
plt.ylabel(r"$\Delta\lambda$ (nm)")
plt.grid(True)
plt.show()

#########################################################################
#########################################################################
''' Calculating and potting surface velocity '''
#########################################################################
#########################################################################
c_in_si=1#3*10**8

surface_velocity=[]
for i in range(len(delay)):
    v=((peak_wavelength[i])**2-lambda_0**2)/((peak_wavelength[i])**2+lambda_0**2)*c_in_si
    surface_velocity.append(v)
    
    
blue_vel=[]
red_vel=[]


for i in range(1,len(peak_wavelength)):
    if(peak_wavelength[i]>peak_wavelength[0]):
        v=((peak_wavelength[i])**2-lambda_0**2)/((peak_wavelength[i])**2+lambda_0**2)*c_in_si
        red_vel.append(v)
    else:
        v=((peak_wavelength[i])**2-lambda_0**2)/((peak_wavelength[i])**2+lambda_0**2)*c_in_si
        blue_vel.append(v)

level=np.ones(len(delta_wavelength))*delta_wavelength[0]
level_index=np.linspace(np.min(delay[1:]),np.max(delay[1:]),len(level))

plt.plot(delay[1:],surface_velocity[1:],'ko-')
plt.plot(blue_delay,blue_vel,'bo')
plt.plot(red_delay,red_vel,'ro')
plt.plot(level_index,level,'k--')
plt.title("critical velocity vs delay")
plt.xlabel("delay (ps)")
plt.ylabel(r"surface velocity in $m\cdot s^{-1}$")
plt.grid(True)
plt.show()


for __var__ in dir():
    exec('del '+ __var__)
    del __var__