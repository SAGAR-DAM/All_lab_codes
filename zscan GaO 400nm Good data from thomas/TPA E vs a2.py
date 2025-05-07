# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:57:23 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

E=np.array([3.5,7,12,20,38,68])
w0=35*10**(-6)
dt=300*10**(-15)
L=600*10**(-6)
lamda=400*10**(-9)
n0=1.0
I=E*10**(-6)/(dt*np.pi*w0**2)
a2=[0.029,0.028,0.017,0.009,0.006,0.004]

la2_err=[0.0042,0.00359,0.0025,0.00116,0.00054,0.00036]
ua2_err=[0.00555,0.00479,0.0035,0.00155,0.00066,0.00045]

e_err=np.array([0.5,1,2,2.9,4,7])
I_err=e_err*10**(-22)/(dt*np.pi*w0**2)
print(E)
print(e_err)
print(I/10**16)
print(I_err)
a2_err=[la2_err,ua2_err]

plt.plot(E,a2,'ro-')
plt.errorbar(E,a2,xerr=e_err,yerr=a2_err,fmt='ko',capsize=4)
plt.xlabel(r"Energy  ($\mu J$)")
plt.ylabel(r"$\alpha_2\ (cm\cdot GW^{-1}$)")
plt.title("Variation of two photon absorption coefficient with"+"\n"+r"input energy on [100] face of $\beta\ Ga_2O_3$")
plt.show()

plt.plot(I/10**16,a2,'ro-')
plt.errorbar(I/10**16,a2,xerr=I_err,yerr=a2_err,fmt='ko',capsize=4)
plt.xlabel(r"$I_0$ (in $TW/cm^2$)",fontsize=12)
plt.ylabel(r"$\alpha_2\ (cm\cdot GW^{-1}$)",fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Variation of two photon absorption coefficient with"+"\n"+r"input intensity on [100] face of $\beta\ Ga_2O_3$")
plt.show()