# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 17:09:24 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.draw import line
from skimage import color
from skimage.transform import resize

# Load the image
image_path ='D:\\Codes\\image processing\\splitting1.png'  # Replace with the actual path to your image
image = imread(image_path)
#image=color.rgba2rgb(image)
image=color.rgb2gray(image)
image=resize(image,(400,400))

def draw_on_image(image,x,y):
    image2=np.copy(image)
    rr, cc = line(x[0],x[1],y[0],y[1])
    line_width = 1
    for i in range(-line_width//2, line_width//2 + 1):
        image2[np.clip(rr + i, 0, image.shape[0] - 1), np.clip(cc + i, 0, image.shape[1] - 1)] = [0]  # Set the color of the line (red in this example)

    #image2[rr, cc] = [1]  # Set the color of the line (red in this example)
    return(image2)

def linecut_function(start_point,end_point,image,image2):
    # Use the numpy function linspace to get the coordinates of the line
    num=round(np.sqrt((start_point[0]-end_point[0])**2+(start_point[1]-end_point[1])**2))
    x, y = np.linspace(start_point[0], end_point[0], num), np.linspace(start_point[1], end_point[1], num)
    image2=draw_on_image(image2, start_point,end_point)
    # Get the grayscale values along the line
    gray_values = image[x.astype(int),y.astype(int)]
    linecut=[]
    for i in range(len(gray_values)):
        linecut_value=gray_values[i]
        linecut.append(linecut_value)
        
    return(np.array(linecut),image2)
    
# Define the points
x = (3*image.shape[0]//8,3*image.shape[1]//8)
y = (5*image.shape[0]//8,5*image.shape[1]//8)

X=image.shape[0]-1
Y=image.shape[1]-1


image2=draw_on_image(image, x, y)
plt.imshow(image2)
plt.axis("off")

no_of_points=3

points_x=np.linspace(x[0],y[0],no_of_points)
points_y=np.linspace(x[1],y[1],no_of_points)
for i in range(len(points_x)):
    points_x[i]=int(points_x[i])
    points_y[i]=int(points_y[i])


linecut_ones_matrix=[]
for i in range(len(points_x)):
    radius=min([points_x[i],image.shape[0]-points_x[i],points_y[i],image.shape[1]-points_y[i]])-1
    theta_degree=np.linspace(0,90,7)    # angels, for which the linecuts will be drawn
    theta=theta_degree*np.pi/180          # angels in radian
    linecut_ones=[]
    for j in range(len(theta)):
        x1=round(points_x[i]+radius*np.sin(theta[j]))
        y1=round(points_y[i]-radius*np.cos(theta[j]))
        
        x2=round(points_x[i]-radius*np.sin(theta[j]))
        y2=round(points_y[i]+radius*np.cos(theta[j]))
        
        start_point=[x1,y1]
        end_point=[x2,y2]
        
        #image2=draw_on_image(image2,start_point,end_point)
        linecut,image2=linecut_function(start_point,end_point,image,image2)
        #plt.plot(linecut)
        #plt.title("pixel no: (%d"%x+",%d"%y+")    theta=%d"%(theta_degree[i]))
        #plt.xlabel('start: %d'%start_point[0]+',%d'%start_point[1]+'\n end: %d'%end_point[0]+',%d'%end_point[1])
        #plt.show()
        
        number_of_ones=len([value for value in linecut if value > 0.5])
        linecut_ones.append(number_of_ones)
    #linecut_ones=np.array(linecut_ones)
    linecut_ones_matrix.append(linecut_ones)
    
plt.imshow(image2)
plt.axis('off')
plt.show()

print(linecut_ones_matrix)
linecut_ones_matrix=np.array(linecut_ones_matrix)
m=np.min(linecut_ones_matrix)
print(m)
b = np.where(linecut_ones_matrix==m)
#print(b)
bpi=b[0][0]
bai=b[1][0]
print(bpi,bai)
print(points_x[bpi],points_y[bpi])
print(theta_degree[bai])

if(points_x[bpi]<=image.shape[0]//2):
    if(theta[bai]<=np.arctan(points_x[bpi]/(Y-points_y[bpi]))):
        x1=int(points_x[bpi]+points_y[bpi]*np.tan(theta[bai]))
        y1=0
        
        x2=int(points_x[bpi]-(Y-points_y[bpi])*np.tan(theta[bai]))
        y2=Y
        
    elif(np.arctan(points_x[bpi]/(Y-points_y[bpi]))<theta[bai]<=np.arctan((X-points_x[bpi])/points_y[bpi])):
        x1=int(points_x[bpi]+points_y[bpi]*np.tan(theta[bai]))
        y1=0
        
        x2=0
        y2=int(points_y[bpi]+points_x[bpi]/np.tan(theta[bai]))
    
    elif(np.arctan((X-points_x[bpi])/points_y[bpi])<theta[bai]<np.pi/2):
        x1=X
        y1=int(points_y[bpi]-((X-points_x[bpi])/np.tan(theta[bai])))
        
        x2=0
        y2=int(points_y[bpi]+points_x[bpi]/np.tan(theta[bai]))
        
    else:
        x1=X
        y1=int(points_y[bpi])
        
        x2=0
        y2=y1
        
elif(points_x[bpi]>image.shape[0]//2):
    if(theta[bai]<=np.arctan((X-points_x[bpi])/points_y[bpi])):
        x1=int(points_x[bpi]+points_y[bpi]*np.tan(theta[bai]))
        y1=0
        
        x2=int(points_x[bpi]-(Y-points_y[bpi])*np.tan(theta[bai]))
        y2=Y
        
    elif(np.arctan((X-points_x[bpi])/points_y[bpi]) < theta[bai] <= np.arctan(points_x[bpi]/(Y-points_y[bpi]))):
        x1=X
        y1=int(points_y[bpi]-((X-points_x[bpi])/np.tan(theta[bai])))
        
        x2=int(points_x[bpi]-(Y-points_y[bpi])*np.tan(theta[bai]))
        y2=Y
        
    elif(np.arctan(points_x[bpi]/(Y-points_y[bpi])) < theta[bai] < np.pi/2):
        x1=X
        y1=int(points_y[bpi]-((X-points_x[bpi])/np.tan(theta[bai])))
        
        x2=0
        y2=int(points_y[bpi]+points_x[bpi]/np.tan(theta[bai]))
        
    else:
        x1=X
        y1=int(points_y[bpi])
        
        x2=0
        y2=y1
        

start_point=[x1,y1]
end_point=[x2,y2]

#image2=draw_on_image(image2,start_point,end_point)
linecut,image2=linecut_function(start_point,end_point,image,image)
plt.imshow(image2)
plt.axis('off')
plt.show()


X=image.shape[0]
Y=image.shape[1]

imagetl = np.asarray([[0]*Y]*X)
imagebr = np.asarray([[0]*Y]*X)

for i in range(X):
    for j in range(Y):
        if(i/np.tan(theta[bai])+j>=points_y[bpi]+points_x[bpi]/np.tan(theta[bai])):
            imagebr[i,j]=image[i,j]
        else:
            imagetl[i,j]=image[i,j]
            
plt.imshow(image)
plt.axis('off')
plt.show()

plt.imshow(imagetl)
plt.axis('off')
plt.show()

plt.imshow(imagebr)
plt.axis('off')
plt.show()

print(image)