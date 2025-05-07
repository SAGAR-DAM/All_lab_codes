# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 11:41:38 2024

@author: mrsag
"""

import numpy as np
import re 
import os

t = '8.33'
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
    
ellipticity = np.zeros(shape=(201,201,101))
for i in range(len(files)):
    f=open(files[i],'r')
    r=np.loadtxt(f)
    
    data=r[:,:]
    ellipticity[:,:,i]=data
    
ellipticity = ellipticity.flatten()

np.savetxt(f"D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\t_{t}\\ellipticity_matrix_t_{t}.txt", ellipticity, fmt='%f', delimiter='\t')
    
# Directory containing the tan_psi files
directory = f'D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\t_{t}\\tan_psi'

# Get list of filenames sorted numerically
files = sorted(os.listdir(directory), key=numeric_sort_key)


# Now you can iterate over the files in increasing numerical order
for i in range(len(files)):
    full_path = os.path.join(directory, files[i])
    files[i] = full_path
    #print(files[i])
    
faraday_rot = np.zeros(shape=(201,201,101))
for i in range(len(files)):
    f=open(files[i],'r')
    r=np.loadtxt(f)
    
    data=r[:,:]
    faraday_rot[:,:,i]=data
    
faraday_rot = faraday_rot.flatten()

np.savetxt(f"D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\t_{t}\\tan_psi_matrix_t_{t}.txt", faraday_rot, fmt='%f', delimiter='\t')