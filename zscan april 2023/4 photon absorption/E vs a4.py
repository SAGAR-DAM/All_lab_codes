# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:57:23 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

w0=35*10**(-6)
dt=300*10**(-15)
L=600*10**(-6)
lamda=400*10**(-9)
n0=1.0
E=np.array([12,20.3,52.8])
I=E*10**(-6)/(dt*np.pi*w0**2)

a4=[6.6,7.7,7.9]

la4_err=[1.1,0.9,1.0]
ua4_err=[1.4,1.2,1.3]

e_err=np.array([0.74,0.912,2.52])
I_err=e_err*10**(-22)/(dt*np.pi*w0**2)
print("E",E)
print("e_err",e_err)
print("I",I/10**16)
print("I_err",I_err)
a4_err=[la4_err,ua4_err]

plt.plot(E,a4,'ro-')
plt.errorbar(E,a4,xerr=e_err,yerr=a4_err,fmt='ko',capsize=4)
plt.xlabel(r"Energy  ($\mu J$)")
plt.ylabel(r"$\alpha_4\ (\times 10^{-34})\ \ cm^5\cdot GW^{-3}$")
plt.title("Variation of 4 photon absorption coefficient with"+"\n"+r"input energy on [100] face of $\beta\ Ga_2O_3$")
plt.show()

plt.plot(I/10**16,a4,'ro-')
plt.errorbar(I/10**16,a4,xerr=I_err,yerr=a4_err,fmt='ko',capsize=4)
plt.xlabel(r"Energy  ($\mu J$)")
plt.ylabel(r"$\alpha_4\ (\times 10^{-34})\ \ cm^5\cdot GW^{-3}$")
plt.title("Variation of 4 photon absorption coefficient with"+"\n"+r"input energy on [100] face of $\beta\ Ga_2O_3$")
plt.show()