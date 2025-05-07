# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:24:13 2024

@author: mrsag
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import glob
from Curve_fitting_with_scipy import Gaussianfitting as Gf
from scipy.signal import fftconvolve
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 10
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

# %%

c = 0.3   #in mm/ps

def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index

def moving_average(signal, window_size):
    # Define the window coefficients for the moving average
    window = np.ones(window_size) / float(window_size)
    
    # Apply the moving average filter using fftconvolve
    filtered_signal = fftconvolve(signal, window, mode='same')
    
    return filtered_signal


def point_avg(arr,n):
    arr1=[]
    for i in range(int(len(arr)/n)):
        x=np.mean(arr[n*i:n*(i+1)])
        arr1.append(x)
    arr1.append(np.mean(arr[(int(len(arr)/n))*n:]))
    
    return(arr1)


def fill_zeros_with_neighbor_avg(arr, max_iterations=1000, tol=1e-10):
    arr = arr.copy()
    for _ in range(max_iterations):
        prev_arr = arr.copy()
        for i in range(len(arr)):
            if arr[i] == 0:
                # Find nearest non-zero to the left
                left = i - 1
                while left >= 0 and arr[left] == 0:
                    left -= 1

                # Find nearest non-zero to the right
                right = i + 1
                while right < len(arr) and arr[right] == 0:
                    right += 1

                left_val = arr[left] if left >= 0 else None
                right_val = arr[right] if right < len(arr) else None

                if left_val is not None and right_val is not None:
                    arr[i] = (left_val + right_val) / 2
                elif left_val is not None:
                    arr[i] = left_val
                elif right_val is not None:
                    arr[i] = right_val
                # if both are None (all zeros), do nothing

        # Break if the array stops changing
        if np.allclose(arr, prev_arr, atol=tol):
            break

    return arr


def fill_zeros_recursively(arr):
    arr = arr.copy().astype(float)
    n = len(arr)
    i = 0

    while i < n:
        if arr[i] != 0:
            start = i
            i += 1
            while i < n and arr[i] == 0:
                i += 1
            end = i
            if end < n and end - start > 1:
                left = arr[start]
                right = arr[end]
                gap = end - start - 1
                temp = [0] * gap
                for step in range((gap + 1) // 2):
                    val = (left + right) / (2 ** (step + 1))
                    temp[step] = val
                    temp[gap - 1 - step] = val
                arr[start + 1:end] = temp
        else:
            i += 1
    return arr



# %%


# files = glob.glob(r"D:\data Lab\400 vs 800 doppler experiment\400 pump 400 probe\400nm Pump spectrum\Original Spectrum\100%150TW\*.txt")
files = glob.glob(r"D:\data Lab\400 vs 800 doppler experiment\400 pump 400 probe\400nm Pump spectrum\Original Spectrum\20TW\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run8_30%_20TW_ret_11-15_250fs\\*.txt")


minw = 400
maxw = 420

f = open(files[0])
r=np.loadtxt(f,skiprows=17,comments='>')

w = r[:,0]
minw = find_index(w, minw)
maxw = find_index(w, maxw)

I = np.zeros(len(w[minw:maxw]))
err = np.zeros(len(I))

index = 0

for i in range(1,len(files),2):
    f = open(files[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    # wavelength = r[:,0]
    intensity = r[:,1]
    
    if(np.max(intensity)>1000):
        intensity -= np.mean(intensity[0:200])
        intensity /= max(intensity)
        
        # wavelength = wavelength[minw:maxw]
        intensity = intensity[minw:maxw] 
        I += intensity
        index += 1 
        for j in range(len(err)):
            err[j] = max([err[j],intensity[j]])

print("files: ",index)
I /= index
err = err-I

fit_I,parameters,string = Gf.Gaussfit(w[minw:maxw], I)

plt.figure(figsize=(4,2))
plt.fill_between(w[minw:maxw], y1=I+err, y2=I-err, color="k",alpha=0.5)
plt.plot(w[minw:maxw], I, 'r-', label = "data",lw=0.5)

plt.plot(w[minw:maxw],fit_I,'b-')
plt.title("Pump spectrum")
#plt.xlim(400,420)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized intensity")
plt.xlim(408,417)

plt.show()  
