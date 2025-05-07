# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 18:57:13 2023

@author: mrsag
"""
"""
This is the fitting function for Gaussian data. The inputs are two same length array of datatype float.
There are 3 outputs:
1. The array after fitting the data. That can be plotted.
2. The used parameters set as the name parameters.
3. The string to describe the parameters set and the fit function.
"""

import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
from decimal import Decimal
import glob 
import pandas as pd

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

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

'''
x=np.linspace(0,20,201)      #data along x axis
y=10*np.exp(-(x-2.564)**2/5)             #data along y axis
random_noise=np.random.uniform(low=-1,high=1,size=(len(y)))
y=y+random_noise

fit_y,parameters,string=Gaussfit(x,y)
plt.plot(x,y,color='k')
plt.plot(x,fit_y)
plt.show()

print(*parameters)
print('FWHM= ', 2.355*parameters[0])
'''


def main():
    files=glob.glob("D:\\data Lab\\24th march Qsmart pulseweidth\\data\\*.csv")
    for i in range(len(files)):
        data=pd.read_csv(files[i])

        x=data["Model"]
        x=x[20:]
        x=x.to_numpy()
        x=np.array(x)
        x=x.astype(np.float64)

        y=data["DPO4104B"]
        y=y[20:]
        y=y.to_numpy()
        y=np.array(y)
        y=y.astype(np.float64)

        fit_y,parameters,string=Gaussfit(x*10**8,y)
        filename=files[i].replace("D:\\data Lab\\24th march Qsmart pulseweidth\\data\\","")
        filename=filename.replace(".csv","")

        plt.figure()
        plt.plot((x-min(x))/(1*10**-9),y,'ro-',markersize=2,label="raw data")
        plt.plot((x-min(x))/(1*10**-9),fit_y,label="Gaussfit")
        plt.title(filename)
        plt.xlabel("time (ns)\n"+string+"\n"+"FWHM: %f"%(10*parameters[0]*2.355)+"ns")
        plt.ylabel("Intensity")
        plt.grid()
        plt.legend()
        plt.savefig("D:\\data Lab\\24th march Qsmart pulseweidth\\Plots\\%s.jpg"%filename,bbox_inches="tight")
        plt.show()

        print(x)
        print(y)



if __name__=="__main__":
    main()