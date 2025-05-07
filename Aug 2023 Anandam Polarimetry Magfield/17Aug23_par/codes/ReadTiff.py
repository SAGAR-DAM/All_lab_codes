# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:29:55 2023

@author: Anandam
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import zoom
import tifffile as tif
import os

def zoom_custom(image, zoom_factor):
    row, col = image.shape
    
    zoomed_image = zoom(image, zoom_factor)
    nrow, ncol = zoomed_image.shape

    # Padding or Cropping
    if nrow > row:
        crop_row = nrow - row
        crop_col = ncol - col
        final_image = zoomed_image[crop_row//2:(crop_row-crop_row//2)+row,crop_col//2:(crop_col-crop_col//2)+col]
    elif row > nrow:
        pad_row = row - nrow
        pad_col = col - ncol
        final_image = np.pad(zoomed_image,[(pad_row//2,(pad_row-pad_row//2)),(pad_col//2,(pad_col-pad_col//2))],mode='constant')
    else:
        final_image = zoomed_image
    
    return final_image

def maximize_correlation(image1, image2):
    max_corr = 0
    best_shift = (0, 0)
    best_angle = 0
    best_scale = 1
    
    Norm1 = (image1 - np.mean(image1)) / np.std(image1)

    #Iterate over all possible shifts
    for dx in range(-50,51):
        for dy in range(-20, 21):
            shift_image2 = np.roll(np.roll(image2, dx, axis=1), dy, axis=0)
            Norm2 = (shift_image2 - np.mean(shift_image2)) / np.std(shift_image2)
            correlation= np.corrcoef(Norm1.flatten(), Norm2.flatten())[0,1]
            print(correlation,dx,dy,'\n')
            
            if correlation > max_corr:
                max_corr = correlation
                best_shift = (dx, dy)

    shift_image2 = np.roll(np.roll(image2, best_shift[0], axis=1), best_shift[1], axis=0)
    #Iterate over all possible rotations
    for angle in range(-5, 6):
        rot_image2 = rotate(shift_image2, angle, reshape=False)

        for scale in np.linspace(0.9, 1.1, 11):
            scale_image2 = zoom_custom(rot_image2, scale)  
            Norm2 = (scale_image2 - np.mean(scale_image2)) / np.std(scale_image2)
            correlation= np.corrcoef(Norm1.flatten(), Norm2.flatten())[0,1]
            print(correlation,best_shift,angle,scale,'\n')

            if correlation > max_corr:
                max_corr = correlation
                best_angle = angle
                best_scale = scale
    print(max_corr,best_shift, best_angle, best_scale)
    corr_image2 = zoom_custom(rotate(shift_image2, best_angle, reshape=False),best_scale)            
#    return corr_image2, max_corr, best_shift, best_angle, best_scale
    return corr_image2

#Define path of your working folder
path = 'D:\Experiments\Mag_field_2May2023'
os.chdir(path)
print("Cutternt path :",os.getcwd(),'\n')

# Read the .tif file
Im1 = tif.imread(r"Par_002.tif")
Im2 = tif.imread(r"Par_002m.tif")

#Im2m = rotate(Im2, 45, reshape=False)
#Im2m = zoom_custom(Im2, 1.1)
#Im2m = np.roll(np.roll(Im2, 0, axis=1), 0, axis=0)

#plt.figure()
#plt.imshow(Im2, cmap='jet')
#plt.figure()
#plt.imshow(2*Im2, cmap='jet')
#plt.axis('off')  # Remove axis ticks and labels
#plt.show()

#Norm1 = (Im1 - np.mean(Im1)) / np.std(Im1)
#Norm2 = (Im2m - np.mean(Im2m)) / np.std(Im2m)
#c=np.corrcoef(Norm1.flatten(), Norm2.flatten())[0,1]
#print(c)

#Im3, max_corr, bestshift, bestangle, bestscale = maximize_correlation(Im1, Im2)
#Im2 = maximize_correlation(Im1, Im2)
#print('\n')
#print(max_corr)
#print(bestshift)
#print(bestangle)
#print(bestscale)
#
Norm1 = (Im1 - np.mean(Im1)) / np.std(Im1)
Norm2 = (Im2 - np.mean(Im2)) / np.std(Im2)
c=np.corrcoef(Norm1.flatten(), Norm2.flatten())[0,1]
print(c)