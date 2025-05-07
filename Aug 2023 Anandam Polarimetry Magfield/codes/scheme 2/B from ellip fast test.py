# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:50:53 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

image=np.random.uniform(low=-1,high=1,size=(100,100))

plt.imshow(image)
plt.colorbar()
plt.title("Ellipticity")
plt.show()

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

def give_nearest_B(final_ellip_array,ellip,old_index):
    ellip_nearest,index=closest_index(final_ellip_array,ellip,old_index)
    
    Bx_nearest=Bx[index[0]][index[1]]
    By_nearest=By[index[0]][index[1]]
    
    return Bx_nearest,By_nearest,index

file_name = "D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\scheme 2\\Fast algo matrices\\B matrices\\Bx.txt"
r = np.loadtxt(file_name, dtype=float, delimiter='\t')
Bx=r[0:201,0:26]
Bx=np.flip(Bx,axis=0)
Bx=np.flip(Bx,axis=1)
Bx[0:100,:]=np.flip(Bx[0:100,:])
Bx=np.roll(Bx,(101,0),axis=(0,1))
#Bx=make_full(Bx)



print(Bx.shape)

file_name = "D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\scheme 2\\Fast algo matrices\\B matrices\\Bx.txt"
r = np.loadtxt(file_name, dtype=float, delimiter='\t')
By=r[0:201,0:26]
By=np.flip(By,axis=0)
By=np.flip(By,axis=1)
By[0:100,:]=np.flip(By[0:100,:])
By=np.roll(By,(101,26),axis=(0,1))
By=np.flip(By,axis=1)
#By=make_full(By)

print(By.shape)


file_name="D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\scheme 2\\Fast algo matrices\\e matrices\\e0005_t_5_0.txt"
r = np.loadtxt(file_name, dtype=float, delimiter='\t')

index=[Bx.shape[0]//2,Bx.shape[1]//2]
r = np.loadtxt(file_name, dtype=float, delimiter='\t')
final_ellip_array=r[0:201,0:26]
final_ellip_array=np.flip(final_ellip_array,axis=0)
final_ellip_array=np.flip(final_ellip_array,axis=1)
final_ellip_array[0:100,:]=np.flip(final_ellip_array[0:100,:])
final_ellip_array=np.roll(final_ellip_array,(101,0),axis=(0,1))
#final_ellip_array=make_full(final_ellip_array)

plt.imshow(Bx)
plt.show()

plt.imshow(By)
plt.show()

plt.imshow(Bx**2+By**2)
plt.show()

plt.imshow(final_ellip_array)
plt.show()

print(final_ellip_array.shape)

output_Bx=np.zeros(shape=image.shape)
output_By=np.zeros(shape=image.shape)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        ellipticity=image[i,j]
        Bx_now,By_now,idx=give_nearest_B(final_ellip_array, ellipticity, index)
        #print(idx)
        output_Bx[i,j]=Bx_now
        output_By[i,j]=By_now
        
        
plt.imshow(output_Bx)
plt.title("Bx")
plt.colorbar()
plt.show()

plt.imshow(output_By)
plt.colorbar()
plt.title("By")
plt.show()

plt.imshow(abs(output_Bx+1j*output_By))
plt.colorbar()
plt.title("Abs(B)")
plt.show()

print(np.max(output_Bx))
#print(np.uint(By))