# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:39:15 2023

@author: sagar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import glob

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

files=glob.glob("D:\\data Lab\\zscan april 2023\\4 photon absorption\\*.csv")


#Crystal information
L=600*10**(-6)
w0=35*10**(-6)
E=np.array([20.3,12-0.74,52.8])*10**(-6)
dt=300*10**(-15)
I=E/(dt*np.pi*w0**2)      # 52.8: 1.436*10**17     20.3: 5.523*10**16       12: 3.265*10**16
n0=1
lamda=800*10**(-9)

#zR=np.pi*w0**2*n0/lamda
zR_array=np.array([1.14,2.14,1.14])*10**(-3)    # 52.8: 1.14         20.3: 1.14      12: 2.14


a3_array=np.arange(15,19,0.1)*10**(-29)


a3_uerr=np.array([1.2,1.5,1.9])
a3_lerr=np.array([1.1,1.0,1.1])

filenumber=1


def moving_avg(arr,n):
    window_size = n
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
    
    # Loop through the array t o
    #consider every window of size 3
    while i < len(arr) - window_size + 1:
    
    	# Calculate the average of current window
    	window_average = round(np.sum(arr[
    	i:i+window_size]) / window_size, 2)
    	
    	# Store the average of current
    	# window in moving average list
    	moving_averages.append(window_average)
    	
    	# Shift window to right by one position
    	i += 1
    return(moving_averages)




def w(z):
    value=w0*np.sqrt(1+(z/zR)**2)
    return value

def R(x,z):
    value=np.log(np.sqrt(1+2*a3*L*(I0*w0**2*x/(w(z))**2)**2)+np.sqrt(2*a3*L*(I0*w0**2*x/(w(z))**2)**2))
    return value

def integrand(x,z):
    value=R(x,z)/(x*np.sqrt(abs(np.log(x)))) 
    return(value)

def T(z):
    value=1/(np.sqrt(np.pi)*(2*a3*L*((I0*w0**2/(w(z))**2)**2)**(1/2)))*integrate.quad(integrand,0,1,args=z)[0]
    return(value)

T=np.vectorize(T)


data=pd.read_csv(files[filenumber])
print(data)


x=data["CH1 - PK2Pk"]
#x=x*3000/max(x)
y=data["CH4 - PK2Pk"]

x=x/y*y[0]
x=x/max(x)
x=x/np.mean(x[0:100])
x=moving_avg(x,10)

I0=I[filenumber]
zR=zR_array[filenumber]
position=np.linspace(0,30,len(x))
xmin_index=(list(x)).index(min(x))
position=position-xmin_index/len(x)*max(position)
minval=50
position=position[minval:]-(position[minval]-position[0])
x=x[minval:]
sample_pos=np.linspace(min(position),max(position),100)

chi_sq_list=[]

for i in range(len(a3_array)):
    a3=a3_array[i]
    fit_y=T(sample_pos/1000)
    fit_y=fit_y/max(fit_y)
    
    filename_at_title=files[filenumber][49:]
    alpha3=a3*10**22*10**6
    
    absolute_error=0
    
    for j in range(len(sample_pos)):
        
        data_index=int(j*((len(x)-1))/(len(sample_pos)-1))
        print(sample_pos[j],position[data_index])
        error=(fit_y[j]-x[data_index])**2
        absolute_error+=error
    
    
    plt.plot(position,x,'r-',label="Scan Data")
    plt.plot(sample_pos,fit_y,'k-',label="Fit data")
    plt.grid()
    plt.xlabel("z position in mm"+"\n"+r"$\alpha_3: $ %f"%alpha3+r"$\times 10^{-22} cm^3W^{-2}$"+"        "+"absolute sq error: %f"%(absolute_error))
    plt.ylabel("Normalized transmission T(z)")
    plt.title(r"3 Photon absorption by z-scan of $Ga_2O_3$"+"\n"+r"Input energy: 52.8$\mu J$")
    plt.show()
    
    chi_sq_list.append(absolute_error)

print(chi_sq_list)
min_error_index=chi_sq_list.index(min(chi_sq_list))
a3=a3_array[min_error_index]

fit_y=T(sample_pos/1000)
fit_y=fit_y/max(fit_y)

filename_at_title=files[filenumber][49:]
alpha3=a3*10**22*10**6


plt.plot(position,x,'r-',label="Scan Data")
plt.plot(sample_pos,fit_y,'k-',label="Fit data")
plt.grid()
plt.xlabel("z position in mm"+"\n"+r"$\alpha_3: $ %f"%alpha3+r"$\times 10^{-22} cm^3W^{-2}$"+"        "+"absolute sq error: %f"%(min(chi_sq_list)))
plt.ylabel("Normalized transmission T(z)")
plt.title(r"3 Photon absorption by z-scan of $Ga_2O_3$"+"\n"+r"Input energy: 52.8$\mu J$")
plt.show()


plt.plot(a3_array*10**28,chi_sq_list,'ko-')
plt.title(r"Absolute error for the fitting for different choice of $\alpha_3$")
plt.xlabel(r"$\alpha_3$ values ($\times 10^{-28} cm^3/W^2$)"+"\n"+r"$calculated\ by: Abs\ Error=\sum_i(y_i^{analytical}-data_i^{measured})^2$")
plt.ylabel("Absolute error")
plt.show()