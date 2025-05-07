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
    
ellipticity = np.zeros(shape=(201,201,100))
for i in range(100):
    f=open(files[i],'r')
    r=np.loadtxt(f)
    
    data=r[:,:]
    ellipticity[:,:,i]=data
    
    
Bx, By, Bz = np.mgrid[-100:100:201j,-100:100:201j,1:100:100j]
#ellipticity = ellipticity[49:150,49:150,:]

print(f"Bx shape: {Bx.shape}")
print(f"By shape: {By.shape}")
print(f"Bz shape: {Bz.shape}")
print(f"ellipticity shape: {ellipticity.shape}")

print(f"ellip_min: {np.min(ellipticity)}")
print(f"ellip_max: {np.max(ellipticity)}")

# Create the plot
fig = go.Figure(data=go.Volume(
    x=Bx.flatten(),
    y=By.flatten(),
    z=Bz.flatten(),
    value=ellipticity.flatten(),
    isomin=np.min(ellipticity),  # Lower bound of data range
    isomax=np.max(ellipticity),  # Upper bound of data range
    opacity=0.15,  # Opacity of the plot
    surface_count=25,  # Number of isosurfaces
    colorscale=colors  # Use custom colorscale
    ))
    
# Update plot layout including title and centering the plot
fig.update_layout(
    title="Ellipticity vs (Bx,By,Bz)",  # Update the title
    scene=dict(
        xaxis=dict(title='Bx'),
        yaxis=dict(title='By'),
        zaxis=dict(title='Bz'),
        aspectratio=dict(x=1, y=1, z=1),
        ),
    width=700,
    height=700,  # Adjust height to center the plot vertically
    margin=dict(autoexpand=True),  # Allow the plot to expand to fit the available space
)


# Show the plot
fig.show(renderer="browser")    
