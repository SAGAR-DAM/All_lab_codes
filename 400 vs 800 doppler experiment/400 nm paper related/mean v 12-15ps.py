# -*- coding: utf-8 -*-
"""
Created on Sat May  3 14:57:33 2025

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index


f = open(r"D:\velocity file 3e17.txt")
r=np.loadtxt(f,skiprows=2, delimiter="," ,comments='>')

t = r[:,0]
v = r[:,1]/2

minw = find_index(t,12)
maxw = find_index(t,15)

t=t[minw:maxw]
v=v[minw:maxw]

plt.plot(t,v)
plt.show()

mean = np.mean(v)

print(f"mean: {mean:.2e} cm/s")
