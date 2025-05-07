# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:39:15 2023

@author: sagar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

files=glob.glob("D:\\data Lab\\zscan april 2023\\*.csv")
information=["190 uJ 400nm","20.28 uJ 800nm","11.97 uJ 800nm","52.8 uJ 800nm","100 uJ 400 nm"]


def moving_avg(arr,window_size):
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




for i in range(5):
    data=pd.read_csv(files[i])
    print(data)

    
    x=data["CH1 - PK2Pk"]
    y=data["CH4 - PK2Pk"]
    
    x=x/y*y[0]
    window_size=5
    x=moving_avg(x,window_size)
    
    filename_at_title=files[i][29:]
    t=np.linspace(0,50,len(x))
    
    plt.figure()
    plt.plot(t,x)
    plt.grid()
    plt.title(filename_at_title+"\n"+information[i])
    #plt.savefig("D:\\data Lab\\zscan april 2023\\plot\\%s.jpg"%(filename_at_title[0:len(filename_at_title)-4]+" "+information[i]),bbox_inches="tight")
    plt.show()
    
    
plt.figure()
for i in range(5):
    data=pd.read_csv(files[i])
    print(data)

    
    x=data["CH1 - PK2Pk"]
    y=data["CH4 - PK2Pk"]
    
    x=x/y*y[0]
    x=moving_avg(x,window_size)
    
    filename_at_title=files[i][30:]
    t=np.linspace(0,50,len(x))

    
    plt.plot(t,x,label=information[i])
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.title("Z-Scan for 800 and 400 nm comparisn ")
#plt.savefig("D:\\data Lab\\zscan april 2023\\plot\\%s.jpg"%("All plots"),bbox_inches="tight")
plt.show()

plt.figure()
for i in range(5):
    data=pd.read_csv(files[i])
    print(data)

    
    x=data["CH1 - PK2Pk"]
    y=data["CH4 - PK2Pk"]
    
    x=x/y*y[0]
    x=moving_avg(x,window_size)
    
    filename_at_title=files[i][30:]
    t=np.linspace(0,50,len(x))

    
    plt.plot(t,x/max(x),label=information[i])
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.title("Z-Scan for 800 and 400 nm comparisn (Normalized)")
#plt.savefig("D:\\data Lab\\zscan april 2023\\plot\\%s.jpg"%("All plots"),bbox_inches="tight")
plt.show()