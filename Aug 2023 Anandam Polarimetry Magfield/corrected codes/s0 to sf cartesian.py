# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:55:44 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi']=200 # highres display

#defining all constants in CGS
e=4.8032e-10                        #electron charge
m=9.1094e-28                        #electron mass
c=2.99792458e10                     #speed of light
wavelength=4e-5                     #used probe wavelength
omega=2*np.pi*c/wavelength          #prob angular frequency
nc=omega**2*m/(4*np.pi*e**2)
e_eular=2.718281

    
# def phase_unwrap(arr):
#     arr=np.array(arr)
#     delta=np.max(arr)-np.min(arr)
#     try:
#         n=int(2*np.pi/delta)
#     except:
#         n=1
#     arr=arr*n
#     arr=np.unwrap(arr)
#     arr/=n
#     return(arr)

def from_1st_2nd_to_3rd_4th_quadrant_for_antisymmetric_function(half_image):
    full_image = np.concatenate([half_image, np.flip(np.flip(half_image, axis=0),axis=1)], axis=0)
    full_image = np.delete(full_image, half_image.shape[0], axis=0)
    return(full_image)


def final_stokes(Bx,By):
    global t
    global omega
    global Bz
    
    chi=0.0     #initial chi
    psi=0.0     #initial psi
    
    s0_1=[np.cos(2*chi)*np.cos(2*psi)]
    s0_2=[np.cos(2*chi)*np.sin(2*psi)]
    s0_3=[np.sin(2*chi)]

    s0=np.array([s0_1,s0_2,s0_3])

    L=t*1e-12*6e6 #Plasma total length
    #dz=L/100   #one slab length
    dz=1e-8
    
    z=0
    n_steps=int(L*np.log(100)/dz)
    M1=np.matrix(np.identity(3))
    M2=np.matrix(np.identity(3))
    
    R1=np.matrix(np.identity(3))
    R2=np.matrix(np.identity(3))
    
    
    B=np.sqrt(Bx**2+By**2+Bz**2)
    theta=np.arccos(Bz/B)
    omega_c=e*B/(m*c)
    phi = 0
    

    # ne_arr=[]
    # psi2_arr=[]
    # chi2_arr=[]
    # mu1_arr=[]
    # mu2_arr=[]
    
    for i in range(n_steps):
        ne=nc*np.exp(z/L-np.log(100))
        omega_p=np.sqrt(4*np.pi*ne*e**2/m)
        
        F=2*omega/omega_c*(1-omega_p**2/omega**2)*np.cos(theta)/(np.sin(theta))**2
        mu1_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1+np.sqrt(1+F**2)))
        mu2_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1-np.sqrt(1+F**2)))
        
        mu1=np.sqrt(mu1_sq)
        mu2=np.sqrt(mu2_sq)
        
        N=(omega_p/omega)**2
        D=1-(e/(m*omega*c))**2*((Bx**2+By**2)/(1-N)+Bz**2)
        
        if np.isnan(mu1) or mu1>=1:
            mu1=0
        if np.isnan(mu2) or mu2>=1:
            mu2=0
            
        if(mu2 != 0):

            prefactor=omega_p**2/((mu1+mu2)*c*omega**3*D)
            
            O1=prefactor*(e/(m*c))**2*((Bx**2-By**2)/(1-omega_p**2/omega**2))
            
            O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-omega_p**2/omega**2))
            
            O3=prefactor*2*omega*e*Bz/(m*c)
            
            # O=np.sqrt(O1**2+O2**2+O3**2)
            
            w1=O1*dz/2
            w2=O2*dz/2
            w3=O3*dz/2
            
            
            sw1=np.sin(w1)
            sw2=np.sin(w2)
            sw3=np.sin(w3)
            
            cw1=np.cos(w1)
            cw2=np.cos(w2)
            cw3=np.cos(w3)
            
            R1[0,0]=cw3*cw2
            R1[0,1]=cw3*sw2*sw1-sw3*cw1
            R1[0,2]=sw3*sw1+cw3*sw2*cw1
            
            R1[1,0]=sw3*cw2
            R1[1,1]=cw3*cw1+sw3*sw2*sw1
            R1[1,2]=sw3*sw2*cw1-cw3*sw1
            
            R1[2,0]=-sw2
            R1[2,1]=cw2*sw1
            R1[2,2]=cw2*cw1
            
            
            
            R2[0,0]=cw3*cw2
            R2[1,0]=cw3*sw2*sw1+sw3*cw1
            R2[2,0]=sw3*sw1-cw3*sw2*cw1
            
            R2[0,1]=-sw3*cw2
            R2[1,1]=cw3*cw1-sw3*sw2*sw1
            R2[2,1]=sw3*sw2*cw1+cw3*sw1
            
            R2[0,2]=sw2
            R2[1,2]=-cw2*sw1
            R2[2,2]=cw2*cw1
            
            
            R1=np.matrix(R1)
            R2=np.matrix(R2)
            
            R = np.dot(R1,R2)
            
            M1 = np.dot(R,M1)
            M2 = np.dot(M2,R)
            
        elif (mu2==0.0):
            phi += (mu1*dz*omega/c)
            
        z += dz
        
    s1 = np.dot(M1,s0)
    s1 = s1.reshape((3,1))
    
    A1 = s1[1]/s1[0]      #tan2psi
    B1 = s1[2]           #sin2chi
    
    s2eta = np.sqrt((A1**2+B1**2)/(1+A1**2))
    t2eta = s2eta/np.sqrt(1-s2eta**2)
    
    eta1 = np.arcsin(np.sqrt((A1**2+B1**2)/(1+A1**2)))/2
    phi1 = np.arcsin(B1/s2eta)
    
    if(np.isnan(phi1)):
        phi1=0
    
    eta2 = eta1
    phi2 = phi1-phi
    
    eta3 = eta2
    phi3 = phi2-phi
    
    psi3 = np.arctan(np.tan(2*eta3)*np.cos(phi3))/2
    chi3 = np.arctan(np.sin(2*eta3)*np.sin(phi3))/2
    
    s3_1=[np.cos(2*chi3)*np.cos(2*psi3)]
    s3_2=[np.cos(2*chi3)*np.sin(2*psi3)]
    s3_3=[np.sin(2*chi3)]

    s3=np.array([s3_1,s3_2,s3_3])
    s3=s3.reshape((3,1))
    
    sf = np.dot(M2,s3)
    sf=sf.reshape((3,1))
    
    try:
        val=np.arctan(sf[2][0]/np.sqrt((sf[0][0])**2+(sf[1][0])**2))
        if(np.isnan(val) or abs(val)>1.56979633):
            val=np.pi/2
    except:
        None
        
    return(val/2)

final_stokes=np.vectorize(final_stokes)



t=0.5    # in ps

resolution=10
B_max=80
Bx,By=np.mgrid[-B_max+0.001:B_max+0.001:(2*resolution-1)*1j,0+0.001:B_max+0.001:resolution*1j]

Bz = 1e6

final_ellip_array=np.tan(final_stokes(Bx*10**6,By*10**6))
#final_ellip_array=np.flip(final_ellip_array,axis=0)
#final_ellip_array=from_1st_2nd_to_3rd_4th_quadrant_for_antisymmetric_function(final_ellip_array)

#final_ellip_array=np.flip(final_ellip_array,axis=0)
#final_ellip_array=abs(final_ellip_array)

# M=final_ellip_array.shape[0]
# N=final_ellip_array.shape[1]

# top_right_quadrant=final_ellip_array
# # Reflect horizontally to get the top-left quadrant
# top_left_quadrant = np.flip(top_right_quadrant, axis=1)*(-1)

# # Reflect both horizontally and vertically to get the bottom two quadrants
# bottom_left_quadrant = np.transpose(top_right_quadrant)
# bottom_right_quadrant = np.flip(bottom_left_quadrant, axis=1)*(-1)

# final_ellip_array=np.block([[top_left_quadrant, top_right_quadrant],
#                         [bottom_left_quadrant, bottom_right_quadrant]])

# final_ellip_array = np.delete(final_ellip_array,M, axis=0)
# final_ellip_array = np.delete(final_ellip_array,N, axis=1)

print(final_ellip_array)
print(final_ellip_array.shape)

ellip_image = from_1st_2nd_to_3rd_4th_quadrant_for_antisymmetric_function(np.flip(final_ellip_array.T,axis=0))

print(ellip_image.shape)
plt.figure()
plt.imshow(ellip_image,cmap="jet",origin="upper",extent=[-B_max,B_max,-B_max,B_max])
plt.colorbar()
plt.xlabel("Bx (MG)")
plt.ylabel("By (MG)")
plt.title(r"variation of ellipticity (tan($\chi$))"+"\nfor t="+f"{t} ps,  Bz: {int(Bz/1e6)}  MG")
#plt.savefig(f"/Volumes/Extreme SSD/Experiments/Polarimetry 18Aug2023/python codes/B_chart/t_{t}_Bz_{int(Bz/1e6)}.png",bbox_inches="tight")
plt.show()


#file_name = f"/Volumes/Extreme SSD/Experiments/Polarimetry 18Aug2023/python codes/B_chart/t_{t}_Bz_{int(Bz/1e6)}.txt"
#np.savetxt(file_name, final_ellip_array, fmt='%f', delimiter='\t')