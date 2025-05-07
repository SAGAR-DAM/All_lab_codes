# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 22:18:30 2024

@author: mrsag
"""

import numpy as np
import re 
import os
import matplotlib.pyplot as plt
import time

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

t_start = time.time()


def error(ellipticity, faraday_rot, e0, f0):
    err = np.zeros(shape=ellipticity.shape)
    err = (np.array(ellipticity)-e0)**2 + (np.array(faraday_rot)-f0)**2
    
    return(err)


def minimize_error_index(ellipticity, faraday_rot, e0, f0):
    err = error(ellipticity, faraday_rot, e0, f0)
    min_err = np.min(err)
    
    index = np.where(err==min_err)
    index = np.array([index[0][0], index[1][0], index[2][0]])
    return(index)

# Define a custom sorting key function
def numeric_sort_key(filename):
    # Extract the numeric part using regular expressions
    match = re.search(r'_Bz_(\d+\.\d+)\.txt$', filename)
    if match:
        return float(match.group(1))  # Convert to float for proper numerical sorting
    else:
        return filename

# Directory containing the ellipticity files
directory = r'D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\t_8.33\\ellipticity'

# Get list of filenames sorted numerically
files = sorted(os.listdir(directory), key=numeric_sort_key)


# Now you can iterate over the files in increasing numerical order
for i in range(len(files)):
    full_path = os.path.join(directory, files[i])
    files[i] = full_path
    print(files[i])
    
ellipticity = np.zeros(shape=(201,201,100))
for i in range(len(files)):
    f=open(files[i],'r')
    r=np.loadtxt(f)
    
    data=r[:,:]
    ellipticity[:,:,i]=data
    
    
# Directory containing the tan_psi files
directory = r'D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\t_8.33\\tan_psi'

# Get list of filenames sorted numerically
files = sorted(os.listdir(directory), key=numeric_sort_key)


# Now you can iterate over the files in increasing numerical order
for i in range(len(files)):
    full_path = os.path.join(directory, files[i])
    files[i] = full_path
    print(files[i])
    
faraday_rot = np.zeros(shape=(201,201,100))
for i in range(len(files)):
    f=open(files[i],'r')
    r=np.loadtxt(f)
    
    data=r[:,:]
    faraday_rot[:,:,i]=data
    
    
def make_e(x,y):
    return(0.5*np.exp(-y**2)+0.5*np.sin(x**3+y))

def make_f(x,y):
    return(np.cos(x+y**3))

res = 21
x,y = np.mgrid[-1:1:res*1j, -1:1:res*1j]

e0 = make_e(x, y)
f0 = make_f(x, y)

plt.imshow(e0,cmap="jet")
plt.title("ellipticity map")
plt.colorbar()
plt.show()

plt.imshow(f0, cmap="jet")
plt.title("faraday rot map")
plt.colorbar()
plt.show()

Bx, By, Bz = np.mgrid[-100:100:201j,-100:100:201j,1:100:100j]

Bx_image = np.zeros_like(e0)
By_image = np.zeros_like(e0)
Bz_image = np.zeros_like(e0)

@np.vectorize
def make_B(e0, f0):
    index = minimize_error_index(ellipticity, faraday_rot, e0, f0)
    # print(index)
    
    # print(f"closest ellipticity:  {ellipticity[index[0],index[1],index[2]]}")
    # print(f"closest faraday_rot:  {faraday_rot[index[0],index[1],index[2]]}")
    
    
    
    # print(ellipticity.shape)
    # print(Bx.shape)
    
    # print(f"Bx: {Bx[index[0],index[1],index[2]]}")
    # print(f"By: {By[index[0],index[1],index[2]]}")
    # print(f"Bz: {Bz[index[0],index[1],index[2]]}")
    
    Bx_sol = Bx[index[0],index[1],index[2]]
    By_sol = By[index[0],index[1],index[2]]
    Bz_sol = Bz[index[0],index[1],index[2]]
    
    return Bx_sol, By_sol, Bz_sol

Bx_image, By_image, Bz_image = make_B(e0,f0)

plt.imshow(Bx_image, cmap="jet")
plt.title("Bx map")
plt.colorbar()
plt.show()


plt.imshow(By_image, cmap="jet")
plt.title("By map")
plt.colorbar()
plt.show()

plt.imshow(Bz_image, cmap="jet")
plt.title("Bz map")
plt.colorbar()
plt.show()


print(f"time:  {time.time()-t_start}")