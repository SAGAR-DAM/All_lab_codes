# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 22:18:30 2024

@author: mrsag
"""

import numpy as np
import re 
import os

def error(ellipticity, faraday_rot, e0, f0):
    err = np.zeros(shape=ellipticity.shape)
    err = (np.array(ellipticity)-e0)**2 + (np.array(faraday_rot)-f0)**2
    
    return(err)

def minimize_error_index(ellipticity, faraday_rot, e0, f0):
    err = error(ellipticity, faraday_rot, e0, f0)
    min_err = np.min(err)
    
    index = np.where(err==min_err)
    print(f"min err:  {min_err}")
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
directory = r'D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\t_0.33\\ellipticity'

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
directory = r'D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\t_0.33\\tan_psi'

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
    
    
e0 = 0
f0 = 0

index = minimize_error_index(ellipticity, faraday_rot, e0, f0)
print(index)

#print(f"closest ellipticity:  {ellipticity[index[0],index[1],index[2]]}")
#print(f"closest faraday_rot:  {faraday_rot[index[0],index[1],index[2]]}")

Bx, By, Bz = np.mgrid[-100:100:201j,-100:100:201j,1:100:100j]

print(ellipticity.shape)
print(Bx.shape)

print(f"Bx: {Bx[index[0],index[1],index[2]]}")
print(f"By: {By[index[0],index[1],index[2]]}")
print(f"Bz: {Bz[index[0],index[1],index[2]]}")

# ellip_reshape = ellipticity.flatten()
# np.savetxt("D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\Plot codes\\ellipticity.txt", ellip_reshape, fmt='%f', delimiter='\t')

# ellip_reshape = np.loadtxt(open("D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\Plot codes\\ellipticity.txt","r"))
# ellip_reshape = ellip_reshape.reshape(ellipticity.shape)

# print(ellip_reshape.shape)