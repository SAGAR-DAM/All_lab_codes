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
from Curve_fitting_with_scipy import exp_decay_fit as eft
from scipy.signal import fftconvolve
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
import pandas as pd

import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 29
# mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=500 # highres display

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


files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14th Feb 2024\\Spectrum\\run7\\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run9_70%_20TW_ret_11-15_250fs\\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run8_30%_20TW_ret_11-15_250fs\\*.txt")

delay = np.linspace(9.5,13.5,len(files)//2)-10.5
delay = 2*delay/c
delay = np.around(delay, decimals=3)

peaks = []

# for i in range(1,len(files),2):
#     f = open(files[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     minw = find_index(wavelength, 392)
#     maxw = find_index(wavelength, 400)
    
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
    
    
    minw = find_index(wavelength, 390)
    maxw = find_index(wavelength, 400)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    intensity -= np.mean(intensity[0:50])
    intensity /= max(intensity)
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks.append(parameters[1])
    
    # plt.plot(wavelength, intensity, '-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks)):
    if (peaks[i]<395 or peaks[i]>396):
        try:
            peaks[i] = (peaks[i+1]+peaks[i-1])/2
        except:
            peaks[i] = 395.35

peaks = moving_average(peaks,10)

for i in range(len(peaks)):
    if (peaks[i]<395 or peaks[i]>396):
        try:
            if((peaks[i-1]<395 and peaks[i+1]>396) or (peaks[i+1]<395 and peaks[i+1]>396)):
                peaks[i] = (peaks[i+1]+peaks[i-1])/2
            else:
                peaks[i] = 395.35
        except:
            peaks[i] = 395.35

delay = delay[0:len(peaks)]






# Replace 'your_file.csv' with the path to your CSV file
file_path = r"D:\data Lab\400 vs 800 doppler experiment\400 pump 400 probe\14th Feb 2024\good reflectivity\Wed Feb 14 15_56_19 2024\MeasLog.csv"

# Read the CSV file
df = pd.read_csv(file_path)

pd_signal = np.array(df["CH2 - PK2Pk"])
norm_factor = np.array(df["CH1 - PK2Pk"])
norm_signal = pd_signal/norm_factor

delay_reflectivity = np.linspace(9.5,13.5,len(pd_signal))-10.5
delay_reflectivity = 2*delay_reflectivity/c
delay_reflectivity = np.around(delay_reflectivity, decimals=3)

maxw = find_index(delay_reflectivity,18)
shift = 10

# %%

t = delay_reflectivity[2:maxw]
sig = (norm_signal/np.max(norm_signal))[2+shift:maxw+shift]

minw1 = find_index(sig,max(sig)) 
maxw1 = find_index(t, 10)

t = t[minw1:maxw1]
sig = sig[minw1:maxw1]

fit_sig, parameters = eft.fit_exp_decay(t, sig)

# %%



# Create the figure and the first axis
fig, ax1 = plt.subplots(figsize=(12, 10))


# plt.figure(figsize=(18,6))
line1, = ax1.plot((delay[0:len(peaks)])[::2],peaks[::2]-395.3, 'ro',label="Doppler Shift",color="brown",markersize=10)
# ax1.errorbar(x=delay[0:len(peaks)::5],y=(peaks-395.3)[::5],yerr=(np.ones(len(peaks))*0.052)[::5], color='k', capsize=1, linewidth=0.5)
# ax1.errorbar(x=(delay[0:len(peaks)])[::3],
#               y=(peaks-395.3)[::3],
#               yerr=(np.ones(len(peaks))*0.052)[::3],
#               color='k', capsize=2, linewidth=0.5)

ax1.plot(((delay[0:len(peaks)])[peaks-395.3<=0])[::1],(peaks[peaks-395.3<=0]-395.3)[::1], 'bo',markersize=10)
# ax1.plot(delay[0:len(peaks)],peaks-395.3, 'k-')
ax1.fill_between(delay[0:len(peaks)],peaks-395.3-0.052,peaks-395.3+0.052,color="k",alpha=0.25)
# ax1.title("Doppler shift")
ax1.set_xlabel("probe delay (ps)",fontweight='bold')
ax1.set_ylabel("Doppler Shift (nm)",color="brown",fontweight='bold')
ax1.tick_params(axis='y', labelcolor='brown')

ax1.set_xlim(-5,18)
ax1.set_ylim(-0.15,0.55)
ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
# plt.title("Doppler shift vs delay\n"+fr"Intensity: 3$\times$"+r"$10^{17}$ W/cm$^2$ (p-pol)")
# ax1.grid(lw = 0.5, color = "black")
ax1.axvline(x=0, color='k', linestyle='--', linewidth=1)
ax1.axvline(x=8.5, color='k', linestyle='--', linewidth=1)
ax1.axvline(x=12.1, color='k', linestyle='--', linewidth=1)



# Create the second y-axis sharing the same x-axis
ax2 = ax1.twinx()
line2, = ax2.plot(delay_reflectivity[2:maxw],(norm_signal/np.max(norm_signal))[2+shift:maxw+shift],"go--",label="reflectivity",markersize=10,lw=2)
line3, = ax2.plot(t,fit_sig,"m-",lw=5,label="Exponential fit")
# ax2.grid(lw=0.5,color="k")
ax2.set_ylabel('\nNormalized Probe Reflectivity\n(arb unit)', color='g',fontweight='bold')
# Get current yticks
yticks = ax2.get_yticks()
# Keep only the positive ticks (optionally excluding zero)
positive_ticks = [tick for tick in yticks if tick >= 0]
# Apply only the positive ticks
ax2.set_yticks(positive_ticks)
ax2.tick_params(axis='y', labelcolor='g')

# Use ax1.legend and manually pass both lines
# legend=ax1.legend([line1, line2, line3], ['Doppler shift', 'Reflectivity', "exponential fit"], loc='lower right')
# # Set font weight of each legend text to bold
# for text in legend.get_texts():
#     # text.set_fontweight('bold')
#     text.set_fontstyle('italic')
#     # text.set_fontsize(25)
# legend.get_frame().set_facecolor('lightgrey')  # or 'grey' if you want it darker

# Generate sample data
x = np.linspace(min(delay), max(delay), 100)
y1 = np.ones(len(x))*1.2
y2 = -np.ones(len(x))*0.2


# Custom handler for double dot
class HandlerDoubleDot(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        red_dot = Line2D([x0 + width * 0.3], [y0 + height/2],
                          marker='o', color='brown', linestyle='None',
                          markersize=10, transform=trans)
        blue_dot = Line2D([x0 + width * 0.7], [y0 + height/2],
                          marker='o', color='blue', linestyle='None',
                          markersize=10, transform=trans)
        return [red_dot, blue_dot]

# Dummy handle for the red+blue dots
double_dot = Line2D([], [], linestyle='None')

# Now create the legend with two entries
legend = ax1.legend(
    [double_dot, ax2.lines[0],ax2.lines[1]],   # <- double dot first, then cos(x) plot
    ['Doppler shift', 'Reflectivity', 'Exponential fit'],
    facecolor="lightgray", loc='lower right',
    frameon=True,     # Enable the frame
    edgecolor='black', # Frame color
    handler_map={double_dot: HandlerDoubleDot()}
)

# Style the text
for text in legend.get_texts():
    text.set_fontstyle('italic')


# Set background colors based on y-values
# plt.fill_between(x, y2, where=(y2 <= 0), color='blue', alpha=0.2)
# plt.fill_between(x, y1, where=(y1 > 0), color='red', alpha=0.2)
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major')  # Optional: control size
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

plt.ylim(-0.3,1.1)
plt.title("Simultaneous measurement of Doppler shift \nand probe reflectivity",fontsize=30,fontweight="bold")
plt.setp(ax1.get_yticklabels())
plt.setp(ax2.get_yticklabels())
plt.setp(ax1.get_xticklabels())

# plt.legend()
plt.show()
