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

files=glob.glob("D:\\data Lab\\Single shot SD\\11_oct_2023_SD_Frog\\scan 1\\*.txt")

set_numbers=10*np.ones(33,dtype=int)
pos=np.arange(0,16.5,0.5)
accepted_filenumbers=np.zeros(len(set_numbers))

peak_wavelength=[]
bandweidth=[]

laser_wavelength=400.8721189041251

#########################################################################
#########################################################################
''' scanning through all files and taking maximum'''
#########################################################################
#########################################################################

file_index=0
l_error=[]
u_error=[]

for i in range(len(set_numbers)):
    pw=[]
    bw=[]
    
    for j in range(set_numbers[i]):
        
        f=open(files[file_index],'r')
        r=np.loadtxt(f,skiprows=17,comments=">")
        
        filename=files[file_index][-12:]
        
        minw=700
        maxw=1300
        #print(maxw)
        
        intensity_base=np.mean(r[:,1][0:100])
        
        wavelength=r[:,0][minw:maxw]
        intensity=r[:,1][minw:maxw]-intensity_base
        intensity=intensity/max(intensity)
        
        fit_y,parameters,info_string=Gaussfit(wavelength,intensity)
        chi_square_test_statistic, p_value = chi_sq(fit_y,intensity) 
        RMSE=relative_error(fit_y,intensity)
        relative_RMSE=RMSE/np.max(intensity)
        
        print(parameters)
        print("chi_sq Value: ",chi_square_test_statistic)
        print("RMSE: ",RMSE)
        print("relarive_RMSE: ",relative_RMSE)
        pw.append(parameters[1])
        bw.append(2.355*parameters[0])
        
        plt.figure()
        plt.plot(wavelength,intensity,label="Observed data")
        if(relative_RMSE<=0.1):
            plt.plot(wavelength,fit_y,'g-',label="Gaussian fit")
        else:
            plt.plot(wavelength,fit_y,'r-',label="Gaussian Fit")
        plt.legend()
        plt.title(filename+"\n"+r"%s"%info_string)
        plt.xlabel("Wavelength (nm) \nchi_sq value of fit: %f"%chi_square_test_statistic +"\nRMSE: %f"%RMSE +"     Relative RMSE: %f"%relative_RMSE)
        plt.ylabel("Normalized intensity")
        plt.grid(color='black', linestyle='-', linewidth=0.5)
        plt.savefig("D:\\data Lab\\Single shot SD\\11_oct_2023_SD_Frog\\scan 1\\plots\\%s.png"%filename,bbox_inches='tight')
        plt.show()
        
        file_index+=1
    
    
    pw_new=[]
    for k in range(len(pw)):
        if abs(pw[k]-np.mean(pw))<np.std(pw):
            pw_new.append(pw[k])
            accepted_filenumbers[i]+=1
    pw=pw_new
    peak_wavelength.append(np.mean(pw))
    bandweidth.append(np.mean(bw))
    l_error.append(peak_wavelength[-1]-min(pw))
    u_error.append(-peak_wavelength[-1]+max(pw))


# Get parabolic fit coefficients
a, b, c = parabolic_fit(pos,peak_wavelength)
# Generate fitted curve using the coefficients
fitted_parabola = parabola(pos, a, b, c)
RMSE=relative_error(fitted_parabola,peak_wavelength)
relative_RMSE=RMSE/(max(peak_wavelength)-min(peak_wavelength))

plt.figure()
plt.plot(pos,peak_wavelength,'b-',label="Peak wavelength curve")
plt.errorbar(pos,peak_wavelength,yerr=[l_error,u_error],fmt='b',capsize=4)
plt.plot(pos,peak_wavelength,'ro')
plt.plot(pos,fitted_parabola,'g--',label="Parabolic fit")
plt.xlabel("Height from top (mm)"+"\nRelative RMSE: %f"%relative_RMSE+f"\n accepted files: {accepted_filenumbers}")
plt.ylabel("peak wavelength (nm)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.legend()
plt.title("Spectrum variation along signal (Top - bottom)"+"\nfit: y="+r"$ax^2+bx+c$, with a=%f"%a)
plt.savefig("D:\\data Lab\\Single shot SD\\11_oct_2023_SD_Frog\\scan 1\\plots\\spectrum variation.png",bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(pos,bandweidth,'go-')
plt.plot(pos,bandweidth,'ro')
plt.xlabel("Height from top (mm)")
plt.ylabel("bandweidth (nm)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.title("Bandweidth variation along signal length \n(Top to bottom)")
plt.savefig("D:\\data Lab\\Single shot SD\\11_oct_2023_SD_Frog\\scan 1\\plots\\bandweidth variation.png",bbox_inches='tight')
plt.show()



for __var__ in dir():
    exec('del '+ __var__)
    del __var__