# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:39:15 2023

@author: sagar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
import glob

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

files=glob.glob("D:\\data Lab\\zscan GaO 400nm Good data from thomas\\z Scan 400nm runs good\\*.csv")
#information=["bad","Input: 200 uJ 400 nm ","Input: 20.29 uJ 800 nm","Input: 12.7 uJ","Input: 11.97 uJ","Input: 11.97 uJ","Input: 52.8 uJ",]


w0=35*10**(-6)
#E=12*10**(-6)
E=np.array([12,3.5,7,68,20,38])*10**(-6)
dt=300*10**(-15)
L=600*10**(-6)
lamda=400*10**(-9)
n0=1.0
I=E/(dt*np.pi*w0**2)

a2_values=[]
zR_values=[]
#zR=1.*10**(-3)
#print(zR*10**3)

la2_err=[0.00416,0.00359,0.00252,0.00116,0.00054,0.00036]
ua2_err=[0.00555,0.00479,0.00353,0.00155,0.00066,0.00045]
minval= np.array([0,0,0,-12,25,55])/100

order=[1,2,0,4,5,3]

def point_avg(arr,n):
    arr1=[]
    for i in range(int(len(arr)/n)):
        x=np.mean(arr[n*i:n*(i+1)])
        arr1.append(x)
    arr1.append(np.mean(arr[(int(len(arr)/n))*n:]))
    
    return(arr1)

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

def T(z,a,zR):
    value=1-a/(1+(z/zR*10**3)**2)
    return(value)

def curvefit(z,t):
    parameters, covariance = fit(T, z, t,maxfev=100000)
    fit_y = T(z, *parameters)

    return(fit_y, parameters)


for i in range(len(files)):
    data=pd.read_csv(files[order[i]])
    print(data)

    
    x=data["CH1 - PK2Pk"]
    #x=x*3000/max(x)
    if(i<=2):
        y=data["CH2 - PK2Pk"]
    else:
        y=data["CH4 - PK2Pk"]
        
    x=x/y*y[0]
    x=x/(np.mean(x[0:50]))
    x=np.asarray(x)
    if(i<=2):
        x=point_avg(x,16)
    else:
        x=point_avg(x,24)
    filename_at_title=files[order[i]][73:]
    pos=np.linspace(0,3,len(x))
    xmin_index=(list(x)).index(min(x))
    pos=pos-xmin_index*(pos[1]-pos[0])
    
    
    pos=np.asarray(pos)*10**(-2)
    print(pos)
    fit_y,parameters=curvefit(pos,x)
    
    print(*parameters)
    a2=parameters[0]*2*np.sqrt(2)/(I[order[i]]*L)*10**11
    a2_values.append(a2)
    zR_values.append(parameters[1])
    print(a2)
    
    
    pos=pos*10**2
    plt.figure()
    plt.plot(pos*10-minval[order[i]],x,'ro-',label="Raw data")#,markersize=3)
    plt.plot(pos*10,fit_y,'k-',label="Fit")
    plt.xlabel(r"Z (mm)"+"\n"+r"value from fit: $a_2$: %f"%a2+r"$^{+%.5f}$"%(ua2_err[i])+r"$_{-%.5f}$"%(la2_err[i])+" cm/GW",fontsize=12)
    plt.ylabel(r"$T_{Norm}(Z)$",fontsize=12)
    if i==3:
        plt.ylim([0.6,1.1])
    #plt.grid()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    #plt.title(filename_at_title+"\n"+r"model: T(z)=1-$\frac{\alpha_{2} I_0 L_{eff}}{2\sqrt{2}(1+z^2/z^2_0)}$")
    plt.title(r"Input energy %.1f"%(E[order[i]]*10**6)+r" $\mu J$")
    # plt.savefig("D:\\data Lab\\zscan april 2023\\plot\\%s.jpg"%(filename_at_title[0:len(filename_at_title)-4]),bbox_inches="tight")
    plt.show()
    
    
print("a2_values: ", a2_values)
print("mean a2_values: ",np.mean(a2_values))
print("zR_values: ",zR_values)



plt.figure()
for i in range(len(files)):
    data=pd.read_csv(files[order[i]])
    print(data)

    
    x=data["CH1 - PK2Pk"]
    #x=x*3000/max(x)
    if(i<=2):
        y=data["CH2 - PK2Pk"]
    else:
        y=data["CH4 - PK2Pk"]
    
    x=x/y*y[0]
    x=x/(np.mean(x[0:50]))
    #x=moving_avg(x, 5)
    if(i<=2):
        x=point_avg(x,16)
    else:
        x=point_avg(x,24)
    
    pos=np.linspace(0,3,len(x))
    xmin_index=(list(x)).index(min(x))
    pos=pos-xmin_index*(pos[1]-pos[0])
    
    filename_at_title=files[order[i]][30:]
    
    
    plt.plot(10*pos-minval[order[i]],x,'o-',label=r"I$_0$: %.2f"%(I[order[i]]/10**4/10**12)+r" $TW/cm^2$")#,markersize=3,linewidth=0.7)
    plt.grid()
    plt.legend(fontsize=9)
    plt.xlabel("Z (mm)",fontsize=12)
    plt.ylabel(r"$T_{Norm}(z)$",fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.title("Normalized transmission as a function of crystal position")
# plt.savefig("D:\\data Lab\\zscan april 2023\\plot\\%s.jpg"%("All plots"),bbox_inches="tight")
plt.show()

