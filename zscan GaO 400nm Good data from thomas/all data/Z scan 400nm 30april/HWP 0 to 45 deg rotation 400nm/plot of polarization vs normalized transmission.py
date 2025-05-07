# -*- coding: utf-8 -*-
"""
Created on Sat May 13 11:32:30 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

angle=np.linspace(0,45,10)
transmission7=np.array([0.773,0.786,0.789,0.857,0.734,0.816,0.843,0.929,0.935,0.818])
transmission11=np.array([1.15,1.11,1.14,1.25,0.914,1.03,1.08,1.06,1.19,1.13])

transmission7=transmission7/max(transmission7)
transmission11=transmission11/max(transmission11)

plt.plot(angle, transmission7, 'ro-', label=r"Energy: 7$\mu$J")
plt.plot(angle, transmission11, 'ko-', label=r"Energy: 11$\mu$J")
plt.grid()
plt.legend()
plt.xlabel("half wave plate angle (degree)")
plt.ylabel("Normalized transmission value")
plt.title(r"Light polarization vs normalized transmission"+"\n"+"in z scan experiment for undoped $Ga_2O_3$ face [100]")
plt.show()

