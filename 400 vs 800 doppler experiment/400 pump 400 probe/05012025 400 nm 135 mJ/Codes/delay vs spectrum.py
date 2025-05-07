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
import random

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 18
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
    filtered_signal = fftconvolve(signal, window, mode='valid')
    
    return filtered_signal


def find_w0_std(filepath):
    files = glob.glob(filepath)
    p = []

    
    for i in range(len(files)):
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
        
        base_error = np.max(intensity[:100])#-np.min(intensity[:100])
        for w in wavelength:
            if (w<412 or w>415) and find_index(wavelength,w)>base_error:
                intensity[find_index(wavelength,w)] = random.uniform(-base_error,base_error)
        
        # max_I = max(intensity)
        # intensity = np.array(intensity)
        # intensity[find_index(wavelength,412):find_index(wavelength,415)] /= max_I
        
        fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
        
        
        error = np.std(fit_I-intensity)
        
        if(error<0.5 and max(fit_I)>0.2):
            p.append(parameters[1])
            plt.plot(wavelength, intensity)
            plt.plot(wavelength, fit_I,'-')
            
    plt.title(filepath[85:-5])
    plt.xlim(410,416)
    plt.show()
            
    if(len(p)>0):
        peak_w0 = np.mean(p)
        std_w = np.std(p)
        
    return peak_w0, std_w



pr_only = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\05012025 400 nm 135 mJ\\all delays\\pr only\\*.txt"    

peak_pr_only, std_pr_only = find_w0_std(pr_only)

delays = np.array(sorted([28.9,29.65,29.95,30.4,30.7,30.25,30.55,30.85,31.6,31.15,31.45,31.75,32.65,33.4,34.9,36.4]))
time_delay = (delays-30.4)*2/c

peaks = []
errors = []

for i in range(len(delays)):
    delay = delays[i]
    print(delay)
    filepath = f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\05012025 400 nm 135 mJ\\all delays\\{delay}\\*.txt"
    
    peak, std = find_w0_std(filepath)
    
    peaks.append(peak)
    errors.append(min([std,std_pr_only]))
    
peaks = np.array(peaks)
peak_pr_only = np.mean(peaks[0:3])

# peaks = moving_average(signal=peaks, window_size=1)
# time_delay = moving_average(signal=time_delay, window_size=1)

# plt.figure(figsize=(18,6))
plt.plot(time_delay, peaks-peak_pr_only,'ro')
plt.plot(time_delay, peaks-peak_pr_only,'b-')
plt.title("Doppler shift of probe"+"\n"+r"I$_{400}$: $6\times 10^{18} W/cm^2$ (p pol)")
plt.plot(time_delay[0:len(peaks)],peaks-peak_pr_only, 'ro')
# plt.errorbar(x=time_delay[0:len(peaks)],y=peaks-peak_pr_only,yerr=np.ones(len(peaks))*0.052, color='k', capsize=1, linewidth=0.5)
plt.plot((time_delay[0:len(peaks)])[peaks-peak_pr_only<=0],peaks[peaks-peak_pr_only<=0]-peak_pr_only, 'bo')
# plt.errorbar(time_delay,peaks-peak_pr_only,yerr=errors[1:-1], lw=0, elinewidth=1, capsize=2, color = 'k')
# plt.errorbar(time_delay,peaks-peak_pr_only,yerr=np.ones(len(time_delay))*std_pr_only, lw=0, elinewidth=1, capsize=2, color = 'k')
plt.ylim(1.1*np.min(peaks-peak_pr_only),1.1*np.max(peaks-peak_pr_only))
# plt.xlim(-10,25)
plt.xlabel("delay (ps)")
plt.ylabel("Doppler shift (nm)")

# Generate sample data
x = np.linspace(min(time_delay),max(time_delay), 100)
y1 = np.ones(len(x))*0.05
y2 = np.ones(len(x))*-0.150
# plt.xlim(-12,27)
# Set background colors based on y-values
plt.fill_between(x, y2, where=(y2 <= 0), color='blue', alpha=0.3)
plt.fill_between(x, y1, where=(y1 > 0), color='red', alpha=0.3)
plt.grid(color="k", lw=0.5)
plt.show()



def calc_vel(w, w0):
    c_norm = -1
    v = (w**2-w0**2)/(w**2+w0**2)*c_norm
    return v

lambda0 = peak_pr_only


velocity = calc_vel(np.array(peaks), peak_pr_only)

blue_v = []
red_v = []

blue_delay = []
red_delay= []

for i in range(len(velocity)):
    if(velocity[i]<=0):
        blue_delay.append(time_delay[i])
        blue_v.append(velocity[i])
        
    else:
        red_delay.append(time_delay[i])
        red_v.append(velocity[i])
        
blue_v=np.array(blue_v)
red_v=np.array(red_v)


v_uerr = np.abs(calc_vel(w=peaks+std_pr_only, w0=lambda0)-velocity)
v_lerr = np.abs(velocity-calc_vel(w=peaks-std_pr_only, w0=lambda0))
x_uerr = np.ones(len(time_delay))*1
x_lerr = np.ones(len(time_delay))*1

plt.plot(time_delay,velocity, 'ro')
plt.plot(time_delay,velocity, 'k-')
plt.title("Doppler Velocity")
plt.xlabel("probe delay (ps)")
plt.ylabel("v/c")
# plt.xlim(-5,18)
#plt.ylim(-0.125,0.5)
plt.grid(lw = 0.5, color = "black")
plt.show()


plt.figure(figsize=(18,6))
plt.plot(blue_delay,blue_v, 'ro')
plt.plot(red_delay,red_v, 'bo')
plt.plot(time_delay,velocity, 'k-')
# plt.errorbar(x=time_delay, y=np.array(velocity), yerr=[v_lerr,v_uerr], elinewidth=0.5, capsize=1, capthick=0.5, color = 'k')
#plt.errorbar(x=delay, y=np.array(velocity), xerr=[x_lerr,x_uerr], elinewidth=0.5, capsize=1, capthick=0.5, color = 'k')
plt.title("Doppler Velocity of critical surface"+"\n"+fr"Intensity: 6$\times$"+r"$10^{18}$ W/cm$^2$")
plt.xlabel("""probe delay (ps)""", size = 20)
plt.ylabel(r"$\beta$ (v/c)", size = 20)
# plt.xlim(-12,27)
plt.xticks(size = 20)
plt.yticks(size = 20)
#plt.ylim(-0.125,0.5)


# Generate sample data
x = np.linspace(min(time_delay), max(time_delay), 100)
y1 = np.ones(len(x))*(max(velocity)+max(v_uerr))
y2 = np.ones(len(x))*(min(velocity)-min(v_lerr))

# Set background colors based on y-values
plt.fill_between(x, y2, where=(y2 <= 0), color='red', alpha=0.3)
plt.fill_between(x, y1, where=(y1 > 0), color='blue', alpha=0.3)


plt.grid(lw = 0.5, color = "black")
plt.show()