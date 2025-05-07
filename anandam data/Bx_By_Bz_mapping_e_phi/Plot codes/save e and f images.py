# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 11:41:38 2024

@author: mrsag
"""

import numpy as np
import re 
import os
import matplotlib.pyplot as plt


import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

t = '3.00'
# Define a custom sorting key function
def numeric_sort_key(filename):
    # Extract the numeric part using regular expressions
    match = re.search(r'_Bz_(\d+\.\d+)\.txt$', filename)
    if match:
        return float(match.group(1))  # Convert to float for proper numerical sorting
    else:
        return filename

# Directory containing the ellipticity files
directory = f'D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\t_{t}\\ellipticity'

# Get list of filenames sorted numerically
files = sorted(os.listdir(directory), key=numeric_sort_key)


# Now you can iterate over the files in increasing numerical order
for i in range(len(files)):
    full_path = os.path.join(directory, files[i])
    files[i] = full_path
    #print(files[i])
    
for i in range(len(files)):
    f=open(files[i],'r')
    r=np.loadtxt(f)
    
    data=r[:,:]
    
    r = np.flip(r,axis=0)
    
    plt.figure()
    plt.imshow(np.abs(r), cmap="jet", extent = [-100,100,-100,100])
    plt.suptitle(f"t : {t} ps;  Bz: {i} MG")
    plt.title(r"Ellipticity mod value (|tan $\chi$|) vs $\vec B$")
    plt.xlabel("Bx  (MG)")
    plt.ylabel("By  (MG)")
    plt.colorbar()
    plt.savefig(f"F:\\From Lenovo Thinkpad\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\Plots\\t_{t}\\ellipticity\\ellipticity_t_{t}_Bz_{i}.png",bbox_inches="tight")
    plt.show()
    
    
# Directory containing the tan_psi files
directory = f'D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\t_{t}\\tan_psi'

# Get list of filenames sorted numerically
files = sorted(os.listdir(directory), key=numeric_sort_key)


# Now you can iterate over the files in increasing numerical order
for i in range(len(files)):
    full_path = os.path.join(directory, files[i])
    files[i] = full_path
    #print(files[i])
    

for i in range(len(files)):
    f=open(files[i],'r')
    r=np.loadtxt(f)
    
    data=r[:,:]
    
    r = np.flip(r,axis=0)
    
    plt.figure()
    plt.imshow(np.abs(r), cmap="jet", extent = [-100,100,-100,100])
    plt.suptitle(f"t : {t} ps;  Bz: {i} MG")
    plt.title(r"Faraday rotation (|tan $\psi$|) vs $\vec B$")
    plt.xlabel("Bx  (MG)")
    plt.ylabel("By  (MG)")
    plt.colorbar()
    plt.savefig(f"F:\\From Lenovo Thinkpad\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\Plots\\t_{t}\\faraday rotation\\faraday_rot_t_{t}_Bz_{i}.png",bbox_inches="tight")
    plt.show()
    
# for __var__ in dir():
#     exec("del" + __var__)
#     del __var__
    
# import sys
# sys.exit()
    