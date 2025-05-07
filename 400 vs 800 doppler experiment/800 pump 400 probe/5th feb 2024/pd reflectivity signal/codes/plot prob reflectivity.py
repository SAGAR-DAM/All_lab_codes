# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:24:23 2024

@author: mrsag
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

file = "D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\pd reflectivity signal\\5th Feb2024\\Mon Feb 05 16_43_52 2024\\MeasLog.csv"

f = pd.read_csv(file)

data = np.array(f["CH2 - PK2Pk"])
delay = 2*(np.linspace(11,15,len(data))-12.33)/0.3
print(len(data))

print(data)


plt.plot(delay,data,'ro-')
plt.plot(delay,data,'bo')
plt.grid(color="black", lw=0.5)
plt.title("Refleceted probe vs pump-probe delay"+"\n(on a BK-7 glass target)")
plt.ylabel("Oscilloscope signal (mv)")
plt.xlabel("Probe delay (ps)")
plt.show()
