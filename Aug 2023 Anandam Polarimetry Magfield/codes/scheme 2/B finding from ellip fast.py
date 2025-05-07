# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:46:29 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

import matplotlib
matplotlib.rcParams["figure.dpi"]=300

image=np.random.uniform(low=-0.005,high=-0.005,size=(200,200))
X=image.shape[0]
Y=image.shape[1]
for i in range(X):
    for j in range(Y):
        image[i,j]=np.exp(-0.001*((i-X//2)**2+(j-Y//2)**2))
#image=image/max(image)


def make_full(array):
    M=array.shape[0]
    N=array.shape[1]
    
    array = np.block([[ np.flip(array, axis=1),array],
                           [np.flip(np.flip(array,axis=0),axis=1), np.flip(np.fliplr(array), axis=(0, 1))]])
    
    array = np.delete(array,M, axis=0)
    array = np.delete(array,N, axis=1)
    return array

def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (abs(array - value)).argmin()
    val=array.flat[idx]
    index=np.where(array==val)
    return val,index

def dist_in_array(index1,index2):
    dist=0
    for i in range(len(index2)):
        dist+=(index2[i]-index1[i])**2
    return(dist)
    
def closest_index(array,value,old_index):
    closest_val,closest_val_indices=find_nearest_value(array,value)
    #print(f"val: {val}")
    #print(f"index: {index}")
    dist=[]
    for i in range(len(closest_val_indices[0])):
        index2=[]
        for j in range(len(old_index)):
            index2.append(closest_val_indices[j][i])
        #print(f"index2: {index2}")
        dist.append(dist_in_array(old_index,index2))
    
    mindist,mindist_index=find_nearest_value(dist,np.min(dist))
    #print(f"mindist: {mindist}")
    #print(f"mindist_index: {mindist_index}")
    mindist_index_of_closest_val_in_arr=[]
    
    for i in range(len(old_index)):
        #print(mindist_index[0][0])
        mindist_index_of_closest_val_in_arr.append(closest_val_indices[i][mindist_index[0][0]])
    return(closest_val,mindist_index_of_closest_val_in_arr)

def give_nearest_B(ellip_at_t,ellip,old_index):
    ellip_nearest,index=closest_index(ellip_at_t,ellip,old_index)
    
    Bx_nearest=Bx[index[0]][index[1]]
    By_nearest=By[index[0]][index[1]]
    
    return Bx_nearest,By_nearest#,index


def calculate_B(image,ellip_at_t):
    output_Bx=np.zeros(shape=image.shape)
    output_By=np.zeros(shape=image.shape)
    index=[Bx.shape[0]//2,Bx.shape[1]//2]
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(abs(image[i,j]<=0.001)):
                output_Bx[i,j]=0
                output_By[i,j]=0
            else:
                ellipticity=image[i,j]
                Bx_now,By_now=give_nearest_B(ellip_at_t, ellipticity, index)
                output_Bx[i,j]=Bx_now
                output_By[i,j]=By_now
    return(output_Bx,output_By)


def process_for_Bx(file_name):
    r = np.loadtxt(file_name, dtype=float, delimiter='\t')
    Bx=r[0:201,0:26]
    Bx=np.flip(Bx,axis=0)
    Bx=np.flip(Bx,axis=1)
    Bx[0:100,:]=np.flip(Bx[0:100,:])
    Bx=np.roll(Bx,(101,0),axis=(0,1))
    #Bx=make_full(Bx)
    return(Bx)

def process_for_By(file_name):
    r = np.loadtxt(file_name, dtype=float, delimiter='\t')
    By=r[0:201,0:26]
    By=np.flip(By,axis=0)
    By=np.flip(By,axis=1)
    By[0:100,:]=np.flip(By[0:100,:])
    By=np.roll(By,(101,26),axis=(0,1))
    By=np.flip(By,axis=1)
    #By=make_full(By)
    return(By)

def process_for_ellip_at_t(file_name):
    r = np.loadtxt(file_name, dtype=float, delimiter='\t')
    ellip_at_t=r[0:201,0:26]
    ellip_at_t=np.flip(ellip_at_t,axis=0)
    ellip_at_t=np.flip(ellip_at_t,axis=1)
    ellip_at_t[0:100,:]=np.flip(ellip_at_t[0:100,:])
    ellip_at_t=np.roll(ellip_at_t,(101,0),axis=(0,1))
    #ellip_at_t=make_full(ellip_at_t)
    return(ellip_at_t)

file_name_Bx = "D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\scheme 2\\Fast algo matrices\\B matrices\\Bx.txt"
Bx=process_for_Bx(file_name_Bx)
print(Bx.shape)

file_name_By = "D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\scheme 2\\Fast algo matrices\\B matrices\\Bx.txt"
By=process_for_By(file_name_By)
print(By.shape)


file_name_ellip_at_t="D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\scheme 2\\Fast algo matrices\\e matrices\\e0001_t_1_0.txt"
ellip_at_t=process_for_ellip_at_t(file_name_ellip_at_t)
print(ellip_at_t.shape)


plt.imshow(Bx)
plt.colorbar()
plt.show()

plt.imshow(By)
plt.colorbar()
plt.show()

plt.imshow(np.sqrt(Bx**2+By**2))
plt.colorbar()
plt.show()

plt.imshow(ellip_at_t)
plt.colorbar()
plt.show()

plt.imshow(image)
plt.colorbar()
plt.title("Ellipticity map")
plt.show()


output_Bx,output_By=calculate_B(image,ellip_at_t)

plt.imshow(output_Bx)
plt.title("Bx mapping (in MG)")
plt.colorbar()
plt.show()

plt.imshow(output_By)
plt.colorbar()
plt.title("By mapping (in MG)")
plt.show()

modB=np.array(abs(output_Bx+1j*output_By))
plt.imshow(modB)
plt.colorbar()
plt.title("Abs(B) mapping (in MG)")
plt.show()

#print(modB)
plt.hist(modB.flatten(),bins=100,range=(1,100))
plt.xlabel("|B| in MG")
plt.ylabel("No of pixels")
plt.show()

# lensoffset=0
# xx = yy = zz = np.arange(-200,200,1)
# xy = xz = yx = yz = zx = zy = np.zeros_like(xx)    
# mlab.plot3d(yx,yy+lensoffset,yz,line_width=0.01,tube_radius=1)
# mlab.plot3d(zx,zy+lensoffset,zz,line_width=0.01,tube_radius=1)
# mlab.plot3d(xx,xy+lensoffset,xz,line_width=0.01,tube_radius=1)
# mlab.mesh(Bx,By,25*ellip_at_t)
# mlab.axes(extent=[-100, 100, -100, 100, -40, 40], color=(0, 0, 0), nb_labels=5)