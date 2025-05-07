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

files=glob.glob("D:\\data Lab\\zscan april 2023\\*.csv")
information=["bad","Input: 200 uJ 400 nm ","Input: 20.29 uJ 800 nm","Input: 12.7 uJ","Input: 11.97 uJ","Input: 11.97 uJ","Input: 52.8 uJ",]


#Crystal information
L=600*10**(-4)
I0=4.125*10**12
w0=200*10**(-4)
n0=1
lamda=800*10**(-7)
zR=np.pi*w0**2*n0/lamda
a4=4.5*10**(-32)
print(zR)
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
    value=1/(3*np.sqrt(np.pi)*(3*a4*L*(I0*w0**2/(w(z))**2)**(1/3)))*integrate.quad(integrand,0,1,args=z)[0]
    return(value)


data=pd.read_csv(files[3])
print(data)


x=data["CH1 - PK2Pk"]
x=x*3000/max(x)
y=data["CH4 - PK2Pk"]

x=x/y*y[0]
x=x/max(x)
x=x/np.mean(x[0:100])

position=np.linspace(0,30,len(x))/10
xmin_index=(list(x)).index(min(x))
position=position-xmin_index/len(x)*max(position)

filename_at_title=files[3][30:]

plt.figure()
plt.plot(position,x)
plt.grid()
plt.title(filename_at_title+"\n"+information[3])
plt.savefig("D:\\data Lab\\zscan april 2023\\plot\\%s.jpg"%(filename_at_title[0:len(filename_at_title)-4]+" "+information[3]),bbox_inches="tight")
plt.show()
    
    
plt.figure()

position=np.linspace(-100.5,100,30)

Transmission=[]
for i in range(len(position)):
    trans=T(position[i])
    Transmission.append(trans/10**27)

Transmission=1-np.array(Transmission)/max(Transmission)
plt.plot(position,Transmission)
plt.show()