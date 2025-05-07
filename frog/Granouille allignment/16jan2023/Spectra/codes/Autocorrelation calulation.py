# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:26:56 2023

@author: mrsag
"""

'''
Description:
Put any function in f(x) as the input. The code calculates the autocorrelation 
of the function and plots it along with the function itself as the output.

'''

from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

def f(x):          #function for which the autocorrelation is needed

    #y=np.exp(-abs(x))                                          #e^-|x|
    #y=np.exp(-x**2)                                            #Gaussian
    #y=np.exp(-x**2)+np.exp(-(x-10)**2)                         #Double Gaussian
    #y=np.exp(-x**2)+np.exp(-(x-10)**2)+np.exp(-(x+10)**2)      #Triple Gaussian
    #y= 1/(1+x**2)                                              #Lorentzian
    y= 1/(1+x**2)+1/(1+(x-20)**2)                               #Double Lorentzian
    #y= 1/(1+x**2)+1/(1+(x-10)**2)+1/(1+(x+10)**2)              #Triple Lorentzian
    
    return y


def integrand(x,t):
    y=f(x)*f(x-t)
    return(y)
    
def autocorrelation(t):
    return(quad(integrand, -np.inf, np.inf, args=(t)))

delay=np.linspace(-100, 100,2000)
function=np.zeros(len(delay))
auto_corr_data=np.zeros(len(delay))


for i in range(len(delay)):
    function[i]=f(delay[i])
    auto_corr_data[i]=autocorrelation(delay[i])[0]
    
plt.figure(figsize = (30,20))
plt.plot(delay,function,'r-',label='function',linewidth=5,markersize=20)
plt.plot(delay,auto_corr_data,'g-',label='autocorrelation',linewidth=5,markersize=20)
plt.legend()
plt.grid()
plt.xticks(fontsize=45,color='purple')
plt.yticks(fontsize=45,color='purple')
plt.legend(fontsize=45)
plt.show()