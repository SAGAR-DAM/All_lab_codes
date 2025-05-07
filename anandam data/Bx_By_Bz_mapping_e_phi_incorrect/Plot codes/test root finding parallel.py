# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 22:18:30 2024

@author: mrsag
"""

import numpy as np
import re 
import os
import matplotlib.pyplot as plt
import multiprocessing
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
    #print(files[i])
    
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
    #print(files[i])
    
faraday_rot = np.zeros(shape=(201,201,100))
for i in range(len(files)):
    f=open(files[i],'r')
    r=np.loadtxt(f)
    
    data=r[:,:]
    faraday_rot[:,:,i]=data
    
    
def make_e(x,y):
    return(0.5*np.exp(-10*y**2)+0.5*np.sin(x**3+y))

def make_f(x,y):
    return(0.5*np.exp(-10*y**2)*np.cos(x+y**3))

res = 101
x,y = np.mgrid[-1:1:res*1j, -1:1:res*1j]

e0 = make_e(x, y)
f0 = make_f(x, y)

plt.imshow(e0,cmap="jet")
plt.title("ellipticity (tan $\chi$) map")
plt.colorbar()
plt.show()

plt.imshow(f0, cmap="jet")
plt.title(r"faraday rotation (tan $\psi$) map ")
plt.colorbar()
plt.show()

Bx, By, Bz = np.mgrid[-100:100:201j,-100:100:201j,1:100:100j]

@np.vectorize
def make_B(e0, f0):
    if(np.isnan(e0)==True or np.isnan(f0)==True):
        Bx_sol = 0
        By_sol = 0
        Bz_sol = 0
        
        return Bx_sol, By_sol, Bz_sol
    
    if(abs(e0)<=1e-4 and abs(f0)<=1e-4):
        Bx_sol = 0
        By_sol = 0
        Bz_sol = 0
        
        return Bx_sol, By_sol, Bz_sol
    
    else:
        index = minimize_error_index(ellipticity, faraday_rot, e0, f0)
        
        Bx_sol = Bx[index[0],index[1],index[2]]
        By_sol = By[index[0],index[1],index[2]]
        Bz_sol = Bz[index[0],index[1],index[2]]
        
        return Bx_sol, By_sol, Bz_sol

# Parallelized function to compute segments
def compute_segment(segment):
    e_segment, f_segment = segment
    return make_B(e_segment, f_segment)

def main():

    num_cores = multiprocessing.cpu_count()  # Get the number of CPU cores
    num_segments = num_cores if res >= num_cores else res  # Determine number of segments
    
    # Divide the grid into segments
    segment_size = res // num_segments
    segments = [(e0[:, i:(i+segment_size)], f0[:, i:(i+segment_size)]) for i in range(0,e0.shape[1],segment_size)]

    
    # Create a pool of workers
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Compute segments in parallel
        results = pool.map(compute_segment, segments)

    # Combine results
    Bx_image = np.concatenate([result[0] for result in results], axis=1)
    By_image = np.concatenate([result[1] for result in results], axis=1)
    Bz_image = np.concatenate([result[2] for result in results], axis=1)


    
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
    
    print(Bz_image.shape)
    print(f"time:  {time.time()-t_start}")
    
if __name__ == "__main__":
    main()
    
    