# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:24:13 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from Curve_fitting_with_scipy import Gaussianfitting as Gf
from scipy.signal import fftconvolve

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

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

files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run7_50%_20TW_ret_11-15_250fs\\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run9_70%_20TW_ret_11-15_250fs\\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run8_30%_20TW_ret_11-15_250fs\\*.txt")

peaks = []
delay = np.linspace(11.3,15.3,len(files)//2)-12.3
delay = 2*delay/c
delay = np.around(delay, decimals=3)

# for i in range(1,len(files),2):
#     f = open(files[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     minw = find_index(wavelength, 400)
#     maxw = find_index(wavelength, 420)
    
#     wavelength = wavelength[minw:maxw]
#     intensity = intensity[minw:maxw] 

#     intensity -= np.mean(intensity[0:50])
    
#     fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
#     peaks.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i]}")
#     #plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

for i in range(1,len(files),2):
    f = open(files[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 400)
    maxw = find_index(wavelength, 420)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    intensity -= np.mean(intensity[0:50])
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks)):
    if (peaks[i]<410 or peaks[i]>412):
        try:
            peaks[i] = (peaks[i+1]+peaks[i-1])/2
        except:
            peaks[i] = 410

peaks = moving_average(peaks,5)

for i in range(len(peaks)):
    if (peaks[i]<410 or peaks[i]>412):
        try:
            if((peaks[i+1]<410 and peaks[i+1]>412) and (peaks[i-1]<410 and peaks[i-1]>412)):
                peaks[i] = (peaks[i+1]+peaks[i-1])/2
            else:
                peaks[i] = 410
        except:
            peaks[i] = 410


delay = delay[0:len(peaks)]

plt.plot(delay[0:len(peaks)],peaks, 'ro')
plt.plot(delay[2:len(peaks)-3],peaks[2:-3], 'k-')
plt.title("Doppler shift  1")
plt.xlabel("probe delay (ps)")
plt.ylabel("Peak wavelength (nm)")
#plt.xlim(-5,18)
plt.ylim(410.85,410.95)
plt.grid(lw = 1, color = "black")
plt.show()

###########################################
blue_v = []
red_v = []

blue_delay = []
red_delay= []
lambda0 = 410.935

for i in range(len(peaks)):
    if(peaks[i]<=lambda0):
        blue_delay.append(delay[i])
        blue_v.append(peaks[i]-lambda0)
        
    else:
        red_delay.append(delay[i])
        red_v.append(peaks[i]-lambda0)

plt.plot(blue_delay,blue_v, 'bo')
plt.plot(red_delay,red_v, 'ro')
plt.plot(delay[2:len(peaks)-3],peaks[2:-3]-lambda0, 'k-')
plt.title("Doppler shift for 800 pump")
plt.xlabel("probe delay (ps)")
plt.ylabel("dopper shift (nm)")
plt.xlim(-5,20)
plt.ylim(410.85-lambda0,410.95-lambda0)
plt.grid(lw = 1, color = "black")
plt.show()
###########################################



plt.plot(delay[0:len(peaks)],peaks-410.9375, 'ro')
plt.plot(delay[0:len(peaks)],peaks-410.9375, 'k-')
plt.title("Doppler shift")
plt.xlabel("probe delay (ps)")
plt.ylabel("Peak wavelength (nm)")
#plt.xlim(-5,18)
plt.ylim(-0.08,0.01)
plt.grid(lw = 1, color = "black")
plt.show()


peaks = peaks[find_index(delay,-5):find_index(delay, 18)]
delay = delay[find_index(delay,-5):find_index(delay, 18)]

lambda0 = 410.9375#np.mean(peaks[0:5])


velocity =[]
for i in range(len(peaks)):
    v = (peaks[i]**2-lambda0**2)/(peaks[i]**2+lambda0**2)
    velocity.append(v)
    
blue_v = []
red_v = []

blue_delay = []
red_delay= []

for i in range(len(velocity)):
    if(velocity[i]<=0):
        blue_delay.append(delay[i])
        blue_v.append(velocity[i])
        
    else:
        red_delay.append(delay[i])
        red_v.append(velocity[i])
def calc_vel(w, w0):
    c_norm = 1
    v = (w**2-w0**2)/(w**2+w0**2)*c_norm
    return v



plt.plot(delay,velocity, 'ro')
plt.plot(delay,velocity, 'k-')
plt.title("Doppler Velocity")
plt.xlabel("probe delay (ps)")
plt.ylabel("v/c")
#plt.xlim(-5,18)
#plt.ylim(-0.125,0.5)
plt.grid(lw = 0.5, color = "black")
plt.show()

plt.plot(blue_delay,blue_v, 'bo')
plt.plot(red_delay,red_v, 'ro')
plt.plot(delay,velocity, 'k-')
plt.title("Doppler Velocity of critical surface"+"\n"+"Energy on target: 35 mJ 800 nm (p pol)")
plt.xlabel("probe delay (ps)")
plt.ylabel("v/c")
#plt.xlim(-5,18)
#plt.ylim(-0.125,0.5)
plt.grid(lw = 0.5, color = "black")
plt.show()
