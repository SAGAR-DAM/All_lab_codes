# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 18:57:32 2023

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,5,6,7,8,9])
var=x.var()
print(var)
print(np.mean((x-np.mean(x))**2))
print(x[:2])
print(x[2:])


# Suppose data array represent a time series data
# each element represent data at given time
# Assume time are equally spaced

t=np.linspace(-10, 10,101)
data=np.zeros(len(t))
for i in range(len(t)):
    y=np.exp(-(t[i])**2)
    data[i]=y
    
def acf(time,data):
	mean = np.mean(time)
	var = data.var()
	length = len(data)

	acf_array = []
	for t in np.arange(0,length):
		temp = np.mean((data[:(length-t)] - mean)*(data[t:] - mean))/var
		#temp = np.mean((data[:(length-t)])*(data[t:]))/var

		acf_array.append(temp)

	acf_array = np.array(acf_array)
	return acf_array

autocorrealtion=acf(t,data)
print(t)
print(np.mean(data))
print(autocorrealtion)

plt.plot(t,data)
plt.plot(autocorrealtion)
plt.show()