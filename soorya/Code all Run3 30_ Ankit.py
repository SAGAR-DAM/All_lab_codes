# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:04:03 2024

@author: SOORYA
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

# Define the moving average function
def moving_average(signal, window_size):
    # Define window coefficients for the moving average
    window = np.ones(window_size) / float(window_size)
    # Apply the moving average filter using fftconvolve
    filtered_signal = fftconvolve(signal, window, mode='same')
    return filtered_signal

def calculate_v_over_c(peak_wavelengths, probe_wavelength):  
    return -3e8*((peak_wavelengths**2 - probe_wavelength**2) / (peak_wavelengths**2 + probe_wavelength**2))

# Define a Gaussian function
def gaussian(x, amplitude, mean, stddev,baseline):
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))+baseline

# Set the directory where your 'Run7' folder is located
folder_path = r'D:\TIFR\Doppler+Reflectivity_FS_800nm Ankit\Run3_30%_0.1ps_19.1_21.6'

# Get a list of all text files in the directory
text_files = [f'a_{i:03d}.txt' for i in range(1, 339,2) if i !=295 and i != 73 and i != 329 and i != 283 and i != 299 and i != 251]
#text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# Lists to store peak values for each text document
peak_wavelengths = []
odd_retro_positions = []
5
# Loop through each text file and plot the data
for file_name in text_files:
    file_path = os.path.join(folder_path, file_name)

    # Load data from the text file
    data = np.loadtxt(file_path, skiprows=17, usecols=(0, 1), comments='>')

    # Extract columns (assuming two columns: wavelength and retro positions)
    wl = data[1050:1200, 0]
    c = data[1050:1200, 1]

     #Plot the data
    #plt.figure()
    #plt.plot(wl, c, label=file_name)
    
    # # Set limits for the x-axis (wavelength)
    #plt.xlim(404, 411)

    # # Add labels and legend
    #plt.xlabel('Wavelength')
    #plt.ylabel('Counts(arb units)')
    #plt.legend()
    #plt.title('Plot of Wavelength vs Counts for Each Text Document')
    
    # Fit the data to a Gaussian function
    #initial_guess = [max(c), np.mean(wl), np.std(wl)]
    #params, covariance = curve_fit(gaussian, wl, c, p0=initial_guess, maxfev=3000)
    initial_guess = [np.max(c), wl[np.argmax(c)], 1.0] 
    params, covariance = curve_fit(gaussian, wl, c, p0=[max(c), wl[np.argmax(c)], 1.0, 150], maxfev=80000)
    
    # Plot the Gaussian fit
    #plt.plot(wl, gaussian(wl, *params), label='Gaussian Fit', linestyle='--')
    
    # Extract the peak value from the Gaussian fit parameters
    peak_wavelength = params[1]
    
    # Append peak values to the lists
    peak_wavelengths.append(peak_wavelength)

    # Annotate the plot with the peak value
    annotation_text = f'Peak Wavelength: {peak_wavelength:.2f}'
    #plt.annotate(annotation_text, xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center', fontsize=10, color='red')

# Extract odd retro positions
start_position = 19.1
end_position = 21.6
#odd_retro_positions = np.linspace(start_position, end_position, num=340)[1::2][:-1]
odd_retro_positions = np.linspace(start_position, end_position, num=len(text_files))

# Calculate the delays using the transformation
delays = [2*(pos - 19.625) / 0.3 for pos in odd_retro_positions]

# Calculate the average probe wavelength of the first 6 data points
probe_wavelength = np.mean(peak_wavelengths[:5])
print(f'Average Probe Wavelength: {probe_wavelength:.2f}')

# Calculate and print the delta_lambda values
delta_lambda_values = np.array(peak_wavelengths) - probe_wavelength 
for i, delta_lambda in enumerate(delta_lambda_values):
    print(f'Delta Lambda for Text Document {i + 1}: {delta_lambda:.2f}')
    


# Color the points based on their position relative to the average probe wavelength line
plt.figure()
plt.scatter(delays, peak_wavelengths,s=10, c=np.where(peak_wavelengths > probe_wavelength, 'red', 'darkblue'))
#plt.errorbar(delays, peak_wavelengths, yerr=0.004, capsize=3)
plt.plot(delays, peak_wavelengths,linewidth=0.5)
# Add a horizontal line at y=average probe wavelength
plt.axhline(y=probe_wavelength, color='black', linestyle='--', label='Probe only Wavelength')
plt.text(delays[0]+8, probe_wavelength, f'{probe_wavelength:.2f}', color='black', ha='left', va='bottom')
plt.xlabel('Delay (in ps)')
plt.ylabel('Peak Wavelengths (in nm)')
plt.grid(True)
plt.legend(fontsize=8)
#plt.title('Peak Values of wavelengths for each delay(in ps)')

#***************************

# Define the decimal x-axis ticks
#decimal_ticks = np.arange(0, 10, 0.5)  # Define the positions of the ticks
#decimal_labels = ['%.1f' % tick for tick in decimal_ticks]  # Convert the tick positions to string labels with one decimal place

# Set the x-axis ticks
#plt.xticks(decimal_ticks, decimal_labels)
plt.figure()
# Store coordinates of points
x_coords = []
y_coords = []
colors = []
for delay, delta_lambda in zip(delays, delta_lambda_values):
    color = 'red' if delta_lambda > 0 else 'blue'
    plt.plot(delay, delta_lambda, 'o',markersize=5, color=color)  # Plot points
    x_coords.append(delay)
    y_coords.append(delta_lambda)
    colors.append(color)

# Plot line connecting points
for i in range(len(x_coords) - 1):
    plt.plot(x_coords[i:i+2], y_coords[i:i+2], '-',linewidth=0.5, color='black')    
plt.xlabel('Time Delay (in ps)')
plt.ylabel('Doppler shift (in nm)')
#plt.errorbar(x_coords, y_coords, yerr=0.008789148704413958, fmt='o', label='Data') 
#plt.title('Time dependent Doppler shifts in reflected probe spectra')
plt.grid(True)
plt.show()

#***********************************

#moving_avg of delta_lambda vs time delay
moving_average_lambda = moving_average(peak_wavelengths, 5)

#***************************************
moving_avg_delta_lambda = moving_average(delta_lambda_values, 5)

#plt.plot(delays[2:-2], moving_avg_delta_lambda[2:-2], 'bo-')
for i, (x, y) in enumerate(zip(delays[2:-2], moving_avg_delta_lambda[2:-2])):
    color = 'red' if y > 0 else 'blue'
    plt.plot(x, y, 'o-',markersize=5, color=color)
plt.plot(delays[2:-2], moving_avg_delta_lambda[2:-2], '-', linewidth=0.5,color='black')
#plt.errorbar(x_coords, y_coords, yerr=0.008789148704413958, fmt='.', label='Data') 
plt.xlabel('Time Delay (in ps)')
plt.ylabel('Doppler shift (in nm)')
plt.legend(fontsize=8)
plt.grid(True)
#plt.title('Time dependent Doppler shifts in reflected probe spectra')
plt.show()

#****************************************
#Plot velocity vs delay
v_over_c = calculate_v_over_c(moving_average_lambda, probe_wavelength)
velocity = calculate_v_over_c(moving_average_lambda, probe_wavelength)

delays1 = delays
# Store coordinates of points
x_coords = []
y_coords = []
colors = []


plt.plot(delays[2:-2],v_over_c[2:-2],'ko-')
plt.show()


for delays, v_over_c in zip(delays[2:-2], v_over_c[2:-2]):
    color = 'blue' if v_over_c > 0 else 'red'
    plt.plot(delays, v_over_c, 'o',markersize=5, color=color)  # Plot points
    x_coords.append(delays)
    y_coords.append(v_over_c)
    colors.append(color)

# Plot line connecting points
for i in range(len(x_coords) - 1):
    plt.plot(x_coords[i:i+2], y_coords[i:i+2], '-',linewidth=0.5, color='black')



# Plot v/c versus retro position
#plt.plot(delays, v_over_c, 'bo-')
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().ticklabel_format(style="sci", axis="y", scilimits=(0,0))
plt.xlabel('Time Delay (in ps)')
plt.ylabel('Velocity (m/s)')
#plt.title('velocity of probe critical layer versus Time delay (in ps)')
plt.grid(True)
plt.show()

#moving_avg of velocity vs time delay
# moving_avg_vel = moving_average(v_over_c, 5)
# plt.plot(delays[2:-2], moving_avg_vel[2:-2], 'bo-')
# plt.xlabel('Time Delay (in ps)')
# plt.ylabel('Doppler shift (in nm)')
# plt.legend(fontsize=8)
# plt.title('Time dependent Doppler shifts in reflected probe spectra')
# plt.show()


7


def calculate_velocity(position, time):
    """
    Calculate velocity array as a function of time using numerical differentiation.

    Parameters:
        position (array): Array of position values.
        time (array): Array of corresponding time values.

    Returns:
        velocity (array): Array of velocities corresponding to each time point.
    """
    # Check if the length of position and time arrays are equal
    if len(position) != len(time):
        raise ValueError("Length of position and time arrays must be equal.")

    # Differentiate position with respect to time to get velocity
    velocity = np.gradient(position, time)

    return velocity

# Ensure v_over_c and delays are arrays
velocity = np.array(velocity)
delays1 = np.array(delays1)

# Check the types and dimensions
print("Type of v_over_c:", type(velocity))
print("Type of delays:", type(delays1))
print("Length of v_over_c:", len(velocity))
print("Length of delays:", len(delays1))


accleration = calculate_velocity(np.array(velocity[2:-2]), np.array(delays1[2:-2]))

plt.plot(delays1[2:-2],accleration,'ko-')
plt.xlabel('Time Delay (in ps)')
plt.ylabel('Acceleration  (m/s$^2$)')
plt.grid(True)
plt.show()

moving_avg_acceleration = moving_average(accleration, 5)

# Plot the moving average of acceleration
plt.plot(delays1[2:-2], moving_avg_acceleration, 'ko-')
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().ticklabel_format(style="sci", axis="y", scilimits=(0,0))
plt.xlabel('Time Delay (in ps)')
plt.ylabel('Acceleration  (m/s$^2$)')
plt.grid(True)
plt.show()