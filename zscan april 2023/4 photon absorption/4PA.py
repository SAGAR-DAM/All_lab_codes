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
E=np.array([20.3,12,52.8])*10**(-6)
dt=300*10**(-15)
I=E/(dt*np.pi*w0**2)      # 52.8: 1.436*10**17     20.3: 5.523*10**16       12: 3.265*10**16
n0=1
lamda=800*10**(-9)

#zR=np.pi*w0**2*n0/lamda
zR_array=np.array([1.14,2.14,1.14])*10**(-3)    # 52.8: 1.14         20.3: 1.14      12: 2.14

a4_array=np.array([7.7,6.6,7.9])*10**(-44)


a4_uerr=np.array([1.2,1.4,1.3])
a4_lerr=np.array([0.9,1.1,1.0])

filenumber=0
a4=a4_array[filenumber]

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


def point_avg(arr,n):
    arr1=[]
    for i in range(int(len(arr)/n)):
        x=np.mean(arr[n*i:n*(i+1)])
        arr1.append(x)
    arr1.append(np.mean(arr[(int(len(arr)/n))*n:]))
    
    return(arr1)


def w(z):
    value=w0*np.sqrt(1+(z/zR)**2)
    return value

def D(x,z):
    value=(1+(3*a4*L*(I0*w0**2/(w(z))**2)**3*x**3)**(-1))**(1/3)
    return value

def R(x,z):
    value=np.log(np.sqrt((D(x,z))**2+D(x,z)+1)/(D(x,z)-1))-np.sqrt(3)*np.arctan((2*D(x,z)+1)/np.sqrt(3))+2.7207
    return value

def integrand(x,z):
    return R(x,z)/(x*np.sqrt(abs(np.log(x))))  
  
def T(z):
    value=1/(3*np.sqrt(np.pi)*(3*a4*L*((I0*w0**2/(w(z))**2)**3)**(1/3)))*integrate.quad(integrand,0,1,args=z)[0]
    return(value)
T=np.vectorize(T)


data=pd.read_csv(files[filenumber])
print(data)

I0=I[filenumber]
zR=zR_array[filenumber]
x=data["CH1 - PK2Pk"]
#x=x*3000/max(x)
y=data["CH4 - PK2Pk"]

x=x/y*y[0]
x=x/max(x)
x=x/np.mean(x[0:100])
x=moving_avg(x,10)



position=np.linspace(0,30,len(x))
xmin_index=(list(x)).index(min(x))
position=position-xmin_index/len(x)*max(position)
minval=50
position=position[minval:]-(position[minval]-position[0])
x=x[minval:]

sample_pos=np.linspace(min(position),max(position),100)
fit_y=T(sample_pos/1000)
fit_y=fit_y/max(fit_y)

filename_at_title=files[filenumber][49:]
alpha4=a4*10**10*10**34

plt.figure()


plt.plot(position,x,'r-',label="Scan Data")
plt.plot(sample_pos,fit_y,'k-',label="Fit data")
plt.grid()
plt.xlabel("z position in mm"+"\n"+r"fit for:  $\alpha_4: $ %.1f"%alpha4+r"$^{+%.1f}$"%(a4_uerr[filenumber])+r"$_{-%.1f}$"%(a4_lerr[filenumber])+r"$\times 10^{-34} cm^5W^{-3}$")
plt.ylabel("Normalized transmission T(z)")
plt.title(r"4 Photon absorption by z-scan of $Ga_2O_3$"+"\n"+r"Input energy: %.1f"%(E[filenumber]*10**6)+" $\mu J$")
plt.legend()
plt.show()