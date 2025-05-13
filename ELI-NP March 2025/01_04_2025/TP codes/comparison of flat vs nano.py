# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 11:55:40 2025

@author: mrsag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import scipy.integrate as integrate
# from Curve_fitting_with_scipy import polynomial_fit as pft
import glob
from scipy.interpolate import interp1d

import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=500 # highres display

# %%
def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index

def integrated_counts(Energy,counts):
    arr = []
    for E in Energy:
        true_index = (Energy<=E)
        arr.append(np.sum(counts[true_index]))
    return np.array(arr)


# %%
date = "01.04.2025"
filepath = r"D:\data Lab\ELI-NP March 2025\01_04_2025\separated out TP images\Spectrum file nanowire"
files = glob.glob(rf"{filepath}\*.txt")
files = np.delete(files,2)
# %%
E_min = []
E_max = []
length = []
for i,file in enumerate(files):
    data=np.loadtxt(file,skiprows=1)
    
    energy = data[:,0]
    E_min.append(energy[0])
    E_max.append(energy[-1])
    print(energy[0])
    length.append(len(energy))


for i, file in enumerate(files):
    data = np.loadtxt(file, skiprows=1)
    energy = data[:, 0]
    spectrum = data[:, 1]
    
    minw = find_index(energy,max(E_min))
    maxw = find_index(energy,min(E_max))

    energy = energy[minw:maxw]
    spectrum = spectrum[minw:maxw]

    # plt.plot(energy, spectrum, alpha=0.4)
    
    if i==0:
        energy_common_nano = energy
        spectrum_avg_nano = np.zeros(len(spectrum))
        nano_ulim = np.zeros(len(spectrum))
        nano_llim = np.ones(len(spectrum))*1e12
    for j in range(len(spectrum)):
        nano_ulim[j]=max([nano_ulim[j],spectrum[j]])
        nano_llim[j]=min([nano_llim[j],spectrum[j]])
    spectrum_avg_nano += spectrum
    
spectrum_avg_nano /= len(files)
integrated_nano = integrated_counts(energy_common_nano, spectrum_avg_nano)

fig,ax1 = plt.subplots() 
ax1.plot(energy_common_nano,spectrum_avg_nano,"r-",label=f"nano avg({len(files)})")
# plt.fill_between(x=energy_common_nano, y1=nano_llim,y2=nano_ulim,color="r",alpha=0.3,label="nano band")


# %%
filepath = r"D:\data Lab\ELI-NP March 2025\01_04_2025\separated out TP images\Spectrum file flat"
files = glob.glob(rf"{filepath}\*.txt")

# %%
E_min = []
E_max = []
length = []
for i,file in enumerate(files):
    data=np.loadtxt(file,skiprows=1)
    
    energy = data[:,0]
    E_min.append(energy[0])
    E_max.append(energy[-1])
    print(energy[0])
    length.append(len(energy))


for i, file in enumerate(files):
    data = np.loadtxt(file, skiprows=1)
    energy = data[:, 0]
    spectrum = data[:, 1]
    
    minw = find_index(energy,max(E_min))
    maxw = find_index(energy,min(E_max))

    energy = energy[minw:maxw]
    spectrum = spectrum[minw:maxw]

    # plt.plot(energy, spectrum, alpha=0.4)
    
    if i==0:
        energy_common_flat = energy
        spectrum_avg_flat = np.zeros(len(spectrum))
        flat_ulim = np.zeros(len(spectrum))
        flat_llim = np.ones(len(spectrum))*1e12
    for j in range(len(spectrum)):
        flat_ulim[j]=max([flat_ulim[j],spectrum[j]])
        flat_llim[j]=min([flat_llim[j],spectrum[j]])
    spectrum_avg_flat += spectrum
    
spectrum_avg_flat /= len(files)
integrated_flat = integrated_counts(energy_common_flat,spectrum_avg_flat)

ax1.plot(energy_common_flat,spectrum_avg_flat,"b-",lw=1,label=f'flat avg({len(files)})')
# plt.fill_between(x=energy_common_flat, y1=flat_llim,y2=flat_ulim,color="g",alpha=0.3,label="flat band")

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(1,200)
# plt.ylim(1e6,2e11)
ax1.legend(loc="lower center")
ax1.tick_params(axis='y', labelcolor='purple')
ax1.tick_params(axis='x', labelcolor='red')
ax1.set_xlabel("Energy (MeV)\n"+r"$\frac{S_\text{nano}(\epsilon_\text{peak})}{S_\text{flat}(\epsilon_\text{peak})}=$ "+f"{np.max(spectrum_avg_nano)/np.max(spectrum_avg_flat):.1f}    and    "+r"$\frac{I_\text{nano}(\infty)}{I_\text{flat}(\infty)}=$ "+f"{np.sum(spectrum_avg_nano)/np.sum(spectrum_avg_flat):.1f}",color="red")
ax1.set_ylabel(r"Counts:   $S(\epsilon)=\frac{d^2N_{H^+}}{dE\ \cdot d\Omega }\ \ \text{(in MeV}^{-1}\cdot\text{Srad}^{-1})$", color="purple")
ax1.minorticks_on()
# ax1.grid(which='major', linestyle='-', linewidth=0.8, color='black')
# ax1.grid(which='minor', linestyle='-', linewidth=0.5, color='black')
ax1.set_title("Proton sectrum for flat vs nano\n"+f"(avgd over all good shots); Date: {date}")


ax2 = ax1.twinx()
ax2.plot(energy_common_nano,integrated_nano,"r--",lw=1,label="nano integrated")
ax2.plot(energy_common_flat,integrated_flat,"b--",lw=1,label="flat integrated")
ax2.set_yscale("log")
ax2.tick_params(axis="y",labelcolor="magenta")
ax2.set_ylabel("Integrated Count:  "+r"$I(E)=\int_{E_{low}}^ES(\epsilon)d\epsilon$", color="magenta")
plt.show()


# %%


from Curve_fitting_with_scipy import Linefitting as lft

nano_E1 = 10
nano_E2 = 13
nano_E3 = 20
nano_E4 = 35

nano_energy1 = energy_common_nano[np.argmax(spectrum_avg_nano):find_index(energy_common_nano,nano_E2)]
nano_ln_counts1 = (np.log(spectrum_avg_nano))[np.argmax(spectrum_avg_nano):find_index(energy_common_nano,nano_E2)]

nano_energy2 = energy_common_nano[find_index(energy_common_nano,nano_E3):find_index(energy_common_nano,nano_E4)]
nano_ln_counts2 = (np.log(spectrum_avg_nano))[find_index(energy_common_nano,nano_E3):find_index(energy_common_nano,nano_E4)]

nano_fit_1,nano_parameters1 = lft.linefit(nano_energy1,nano_ln_counts1)
nano_fit_2,nano_parameters2 = lft.linefit(nano_energy2,nano_ln_counts2)

nano_T_fit1 = abs(1/nano_parameters1[0])*3/2  # in MeV
nano_T_fit2 = abs(1/nano_parameters2[0])*3/2  # in MeV

flat_E1 = 10
flat_E2 = 10
flat_E3 = 15
flat_E4 = 35

flat_energy1 = energy_common_flat[np.argmax(spectrum_avg_flat):find_index(energy_common_flat,flat_E2)]
flat_ln_counts1 = (np.log(spectrum_avg_flat))[np.argmax(spectrum_avg_flat):find_index(energy_common_flat,flat_E2)]

flat_energy2 = energy_common_flat[find_index(energy_common_flat,flat_E3):find_index(energy_common_flat,flat_E4)]
flat_ln_counts2 = (np.log(spectrum_avg_flat))[find_index(energy_common_flat,flat_E3):find_index(energy_common_flat,flat_E4)]

flat_fit_1,flat_parameters1 = lft.linefit(flat_energy1,flat_ln_counts1)
flat_fit_2,flat_parameters2 = lft.linefit(flat_energy2,flat_ln_counts2)

flat_T_fit1 = abs(1/flat_parameters1[0])*3/2  # in MeV
flat_T_fit2 = abs(1/flat_parameters2[0])*3/2  # in MeV

plt.plot(energy_common_nano,spectrum_avg_nano,"r-",lw=1)
plt.plot(nano_energy1,np.exp(nano_fit_1),linestyle="--",color="orange",lw=5,alpha=0.8,label=f"nano T_low: {nano_T_fit1:.1f} MeV")
plt.plot(nano_energy2,np.exp(nano_fit_2),linestyle="--",color="magenta",lw=5,alpha=0.5,label=f"nano T_high: {nano_T_fit2:.1f} MeV")


plt.plot(energy_common_flat,spectrum_avg_flat,"b-",lw=1)
plt.plot(flat_energy1,np.exp(flat_fit_1),linestyle="--",color="cyan",lw=5,alpha=0.5,label=f"flat T_low: {flat_T_fit1:.1f} MeV")
plt.plot(flat_energy2,np.exp(flat_fit_2),linestyle="--",color="purple",lw=5,alpha=0.5,label=f"flat T_high: {flat_T_fit2:.1f} MeV")

plt.yscale("log")
plt.xlabel("Energy (MeV)")
plt.ylabel(r"Counts:   $S(\epsilon)=\frac{d^2N_{H^+}}{dE\ \cdot d\Omega }\ \ \text{(in MeV}^{-1}\cdot\text{Srad}^{-1})$")
plt.xlim(0,50)
plt.title("Proton Maxwellian Temperature fitting\n"+f"Date: {date}")
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth=0.8, color='black')
plt.grid(which='minor', linestyle='-', linewidth=0.5, color='black')
plt.legend()
plt.show()