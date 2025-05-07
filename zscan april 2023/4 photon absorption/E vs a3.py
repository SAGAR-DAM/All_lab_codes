# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:57:23 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

E=[12,20.3,52.8]

a3=[1.5,1.6,1.9]

la3_err=[0.13,0.11,0.13]
ua3_err=[0.25,0.18,0.24]

e_err=[0.74,0.912,2.52]

a3_err=[la3_err,ua3_err]

plt.plot(E,a3,'ro-')
plt.errorbar(E,a3,xerr=e_err,yerr=a3_err,fmt='ko',capsize=4)
plt.xlabel(r"Energy  ($\mu J$)")
plt.ylabel(r"$\alpha_3\ (\times 10^{-22})\ \ cm^3\cdot GW^{-2}$")
plt.title("Variation of 3 photon absorption coefficient with"+"\n"+r"input energy on [100] face of $\beta\ Ga_2O_3$")
plt.show()