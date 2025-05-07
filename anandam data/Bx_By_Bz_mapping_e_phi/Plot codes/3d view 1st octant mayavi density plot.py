# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:49:58 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
import re 
import os
import plotly.graph_objects as go
from mayavi import mlab

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

# Define custom colorscale mimicking "jet" colormap
colors = [
    [0, "rgb(0,0,128)"],   # Dark blue
    [0.25, "rgb(0,0,255)"], # Blue
    [0.5, "rgb(0,255,255)"], # Cyan
    [0.75, "rgb(255,255,0)"], # Yellow
    [1, "rgb(255,0,0)"]   # Red
]

# Define a custom sorting key function
def numeric_sort_key(filename):
    # Extract the numeric part using regular expressions
    match = re.search(r'_Bz_(\d+\.\d+)\.txt$', filename)
    if match:
        return float(match.group(1))  # Convert to float for proper numerical sorting
    else:
        return filename

# Directory containing the files
directory = r'D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\t_8.33\\ellipticity'

# Get list of filenames sorted numerically
files = sorted(os.listdir(directory), key=numeric_sort_key)


# Now you can iterate over the files in increasing numerical order
for i in range(len(files)):
    full_path = os.path.join(directory, files[i])
    files[i] = full_path
    print(files[i])
    
ellipticity = np.zeros(shape=(101,101,101))
for i in range(len(files)):
    f=open(files[i],'r')
    r=np.loadtxt(f)
    
    data=r[100:201,100:201]
    ellipticity[:,:,i]=data
    
    
Bx, By, Bz = np.mgrid[0:100:101j,0:100:101j,0:100:101j]
#ellipticity = ellipticity[49:150,49:150,:]

print(f"Bx shape: {Bx.shape}")
print(f"By shape: {By.shape}")
print(f"Bz shape: {Bz.shape}")
print(f"ellipticity shape: {ellipticity.shape}")

print(f"ellip_min: {np.min(ellipticity)}")
print(f"ellip_max: {np.max(ellipticity)}")

# Create physical axes
x = np.linspace(0, 100, 101)
y = np.linspace(0, 100, 101)
z = np.linspace(0, 100, 101)
x, y, z = np.meshgrid(x, y, z, indexing='ij')  # Match shape with potential

# Setup the scalar field for volumetric rendering
src = mlab.pipeline.scalar_field(x, y, z, ellipticity)

# Optional: Adjust vmin/vmax to emphasize low potential regions (e.g., near zero)
vmin = np.min(ellipticity)
vmax = np.max(ellipticity)
vol = mlab.pipeline.volume(src, vmin=vmin, vmax=vmax)

# Add axis, title, colorbar
mlab.axes(xlabel='Bx', ylabel='By', zlabel='Bz')
mlab.colorbar(title="Ellipticity", orientation='vertical')
mlab.title('3D Volume Rendering of Potential')
mlab.view(azimuth=45, elevation=60, distance='auto')

mlab.show()