# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:14:57 2024

@author: mrsag
"""

import numpy as np
import happi
import matplotlib.pyplot as plt
import types 
import glob

import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi'] = 300  # highres display

def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    return index

files = glob.glob("D:\\data Lab\\Meta experiment\\02062024 meta angle characterization\\*.trt")
minw = 700
maxw = 900

index = 0
for i in range(0,10):
    f=open(files[i],'r')
    r=np.loadtxt(f,skiprows=8,delimiter=';')

    rangearr=[find_index(r[:,0],minw),find_index(r[:,0],maxw)]
    w=r[:,0][rangearr[0]:rangearr[1]]
    I=r[:,1][rangearr[0]:rangearr[1]]
    I/= max(I)
    index += 1
    
    plt.plot(w, I)
    plt.title("No Meta")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intenity")
    plt.grid(color='k',lw=0.5)
plt.suptitle(f"Total files: {index}")
plt.show()

index=0
for i in range(11,20):
    f=open(files[i],'r')
    r=np.loadtxt(f,skiprows=8,delimiter=';')

    rangearr=[find_index(r[:,0],minw),find_index(r[:,0],maxw)]
    w=r[:,0][rangearr[0]:rangearr[1]]
    I=r[:,1][rangearr[0]:rangearr[1]]
    I/= max(I)
    index += 1
    
    plt.plot(w, I)
    plt.title(r"Meta with Angle: $\theta=$ 0$^o$")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intenity")
    plt.grid(color='k',lw=0.5)
plt.suptitle(f"Total files: {index}")
plt.show()

index=0
for i in range(21,30):
    f=open(files[i],'r')
    r=np.loadtxt(f,skiprows=8,delimiter=';')

    rangearr=[find_index(r[:,0],minw),find_index(r[:,0],maxw)]
    w=r[:,0][rangearr[0]:rangearr[1]]
    I=r[:,1][rangearr[0]:rangearr[1]]
    I/= max(I)
    index += 1
    
    plt.plot(w, I)
    plt.title(r"Meta with Angle: $\theta=$ 10$^o$")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intenity")
    plt.grid(color='k',lw=0.5)
plt.suptitle(f"Total files: {index}")
plt.show()

index = 0
for i in range(31,40):
    f=open(files[i],'r')
    r=np.loadtxt(f,skiprows=8,delimiter=';')

    rangearr=[find_index(r[:,0],minw),find_index(r[:,0],maxw)]
    w=r[:,0][rangearr[0]:rangearr[1]]
    I=r[:,1][rangearr[0]:rangearr[1]]
    I/= max(I)
    index += 1
    
    plt.plot(w, I)
    plt.title(r"Meta with Angle: $\theta=$ 20$^o$")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intenity")
    plt.grid(color='k',lw=0.5)
plt.suptitle(f"Total files: {index}")
plt.show()

index=0
for i in range(41,50):
    f=open(files[i],'r')
    r=np.loadtxt(f,skiprows=8,delimiter=';')

    rangearr=[find_index(r[:,0],minw),find_index(r[:,0],maxw)]
    w=r[:,0][rangearr[0]:rangearr[1]]
    I=r[:,1][rangearr[0]:rangearr[1]]
    I/= max(I)
    index +=1
    plt.plot(w, I)
    plt.title(r"Meta with Angle: $\theta=$ 30$^o$")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intenity")

    plt.grid(color='k',lw=0.5)
plt.suptitle(f"Total files: {index}")
plt.show()

index =0
for i in range(50,len(files)):
    f=open(files[i],'r')
    r=np.loadtxt(f,skiprows=8,delimiter=';')

    rangearr=[find_index(r[:,0],minw),find_index(r[:,0],maxw)]
    w=r[:,0][rangearr[0]:rangearr[1]]
    I=r[:,1][rangearr[0]:rangearr[1]]
    I/= max(I)
    index +=1
    
    plt.plot(w, I)
    plt.title(r"Meta with Angle: $\theta=$ 0$^o$ set 2")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intenity")
    plt.grid(color='k',lw=0.5)
plt.suptitle(f"Total files: {index}")
plt.show()



__del_vars__ = []
# Print all variable names in the current local scope
print("Deleted Variables:")
for __var__ in dir():
    if not __var__.startswith("_") and not callable(locals()[__var__]) and not isinstance(locals()[__var__], types.ModuleType):
        __del_vars__.append(__var__)
        exec("del "+ __var__)
    del __var__
    
print(__del_vars__)