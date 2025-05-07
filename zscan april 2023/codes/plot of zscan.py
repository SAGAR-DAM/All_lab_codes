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
information=["bad","Input: 200 uJ 400 nm ","Input: 20.29 uJ 800 nm","Input: 12.7 uJ","Input: 11.97 uJ","Input: 11.97 uJ","Input: 52.8 uJ",]

for i in range(7):
    data=pd.read_csv(files[i])
    print(data)

    
    x=data["CH1 - PK2Pk"]
    x=x*3000/max(x)
    y=data["CH4 - PK2Pk"]
    
    x=x/y*y[0]
    
    filename_at_title=files[i][30:]
    
    plt.figure()
    plt.plot(x)
    plt.grid()
    plt.title(filename_at_title+"\n"+information[i])
    plt.savefig("D:\\data Lab\\zscan april 2023\\plot\\%s.jpg"%(filename_at_title[0:len(filename_at_title)-4]+" "+information[i]),bbox_inches="tight")
    plt.show()
    
    
plt.figure()
for i in range(1,7):
    data=pd.read_csv(files[i])
    print(data)

    
    x=data["CH1 - PK2Pk"]
    x=x*3000/max(x)
    y=data["CH4 - PK2Pk"]
    
    x=x/y*y[0]
    
    filename_at_title=files[i][30:]
    
    
    plt.plot(x,label=information[i])
    plt.grid()
    plt.legend()
    plt.title("All plots")
plt.savefig("D:\\data Lab\\zscan april 2023\\plot\\%s.jpg"%("All plots"),bbox_inches="tight")
plt.show()