# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:08:06 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt


import matplotlib
matplotlib.rcParams['figure.dpi']=500 # highres display

#defining all constants in CGS
e=4.8032e-10                        #electron charge
m=9.1094e-28                        #electron mass
c=2.99792458e10                     #speed of light
wavelength=4e-5                     #used probe wavelength
omega=2*np.pi*c/wavelength          #prob angular frequency
nc=omega**2*m/(4*np.pi*e**2)
e_eular=2.718281

    
def phase_unwrap(arr):
    arr=np.array(arr)
    delta=np.max(arr)-np.min(arr)
    n=int(2*np.pi/delta)
    arr=arr*n
    arr=np.unwrap(arr)
    arr/=n
    return(arr)
    
def final_stokes(Bx,By):
    if(abs(Bx)<100e3 or abs(By)<100e3):
        global t
        global omega
        chi=0.0     #initial chi
        psi=0.0     #initial psi
        
        s0_1=[np.cos(2*chi)*np.cos(2*psi)]
        s0_2=[np.cos(2*chi)*np.sin(2*psi)]
        s0_3=[np.sin(2*chi)]

        s0=np.array([s0_1,s0_2,s0_3])
        try:
            val=np.arctan(s0[2][0]/np.sqrt((s0[0][0])**2+(s0[1][0])**2))
            if(np.isnan(val) or abs(val)>1.56979633):
                val=np.pi/2
        except:
            None
            
        return(val/2)
    
    
    else:
        global t
        global omega
        chi=0.0     #initial chi
        psi=0.0     #initial psi
        
        s0_1=[np.cos(2*chi)*np.cos(2*psi)]
        s0_2=[np.cos(2*chi)*np.sin(2*psi)]
        s0_3=[np.sin(2*chi)]
    
        s0=np.array([s0_1,s0_2,s0_3])
    
        L=t*1e-12*6e6 #Plasma total length
        #dz=L/100   #one slab length
        dz=5e-8
        
        Bz=10
        B=np.sqrt(Bx**2+By**2+Bz**2)
        theta=np.arccos(Bz/B)
        
        z=0
        n_steps=int(L*np.log(100)/dz)
        
        M=np.matrix(np.identity(3))
        R1=np.matrix(np.identity(3))
        R2=np.matrix(np.identity(3))
        
        
        for i in range(n_steps):
            #ne=(np.exp(z/L)-1)*nc/(e_eular-1)
            ne=nc*np.exp(z/L-np.log(100))
            omega_p=np.sqrt(4*np.pi*ne*e**2/m)
            omega_c=e*B/(m*c)
            
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
            
            prefactor=omega_p**2/((mu1+mu2)*c*omega**3*D)
            
            O1=prefactor*(e/(m*c))**2*((Bx**2-By**2)/(1-omega_p**2/omega**2))
            
            O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-omega_p**2/omega**2))
            
            O3=prefactor*2*omega*e*Bz/(m*c)
            
            #O=np.sqrt(O1**2+O2**2+O3**2)
            
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
            M=np.dot(R1,np.dot(R2,M))
            
            
            sf=np.array(np.dot(M,s0))
            
            z+=dz
        
        for i in range(n_steps):
            
            #ne=(np.exp(z/L)-1)*nc/(e_eular-1)
            ne=nc*np.exp(z/L-np.log(100))
            omega_p=np.sqrt(4*np.pi*ne*e**2/m)
            omega_c=e*B/(m*c)
            
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
            
            prefactor=omega_p**2/((mu1+mu2)*c*omega**3*D)
            
            O1=prefactor*(e/(m*c))**2*((Bx**2-By**2)/(1-omega_p**2/omega**2))
            
            O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-omega_p**2/omega**2))
            
            O3=prefactor*2*omega*e*Bz/(m*c)
            
            #O=np.sqrt(O1**2+O2**2+O3**2)
            
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
            M=np.dot(R1,np.dot(R2,M))
            
            
            sf=np.array(np.dot(M,s0))
            
            z-=dz
        
        sf=np.array(np.dot(M,s0))
        
        try:
            val=np.arctan(sf[2][0]/np.sqrt((sf[0][0])**2+(sf[1][0])**2))
            if(np.isnan(val) or abs(val)>1.56979633):
                val=np.pi/2
        except:
            None
            
        return(val/2)

final_stokes=np.vectorize(final_stokes)

t=3

Bx,By=np.mgrid[0.001:100.001:51j,0.001:100.001:51j]

final_ellip_array=np.tan(final_stokes(Bx*10**6,By*10**6))
final_ellip_array=np.flip(final_ellip_array,axis=0)
M=final_ellip_array.shape[0]
N=final_ellip_array.shape[1]

top_right_quadrant=final_ellip_array
# Reflect horizontally to get the top-left quadrant
top_left_quadrant = np.flip(top_right_quadrant, axis=1)*(-1)

# Reflect both horizontally and vertically to get the bottom two quadrants
bottom_left_quadrant = np.transpose(top_right_quadrant)
bottom_right_quadrant = np.flip(bottom_left_quadrant, axis=1)*(-1)

final_ellip_array=np.block([[top_left_quadrant, top_right_quadrant],
                        [bottom_left_quadrant, bottom_right_quadrant]])

final_ellip_array = np.delete(final_ellip_array,M, axis=0)
final_ellip_array = np.delete(final_ellip_array,N, axis=1)

print(final_ellip_array)


plt.figure()
plt.imshow(final_ellip_array,cmap="jet",origin="upper",extent=[-100,100,-100,100])
plt.colorbar()
plt.xlabel("Bx (MG)")
plt.ylabel("By (MG)")
plt.title(r"variation of ellipticity (tan($\chi$))"+"\nfor t="+f"{t} ps")
plt.savefig("D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\scheme 2\\plots\\%.2f ps.png"%t,bbox_inches="tight")
plt.show()

#file_name = "D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\scheme 2\\Fast algo matrices\\e matrices cartesian\\e0001_t_1_0.txt"
#np.savetxt(file_name, final_ellip_array, fmt='%f', delimiter='\t')

for __var__ in dir():
    exec('del '+ __var__)
    del __var__
    
import sys
sys.exit()