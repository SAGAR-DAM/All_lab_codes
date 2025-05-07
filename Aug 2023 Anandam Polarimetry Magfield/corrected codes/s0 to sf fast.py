 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:08:06 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import matplotlib
matplotlib.rcParams['figure.dpi']=500 # highres display

t_start=time.time()

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
    try:
        n=int(2*np.pi/delta)
    except:
        n=1
    arr=arr*n
    arr=np.unwrap(arr)
    arr/=n
    return(arr)

def final_stokes(s0,Bx,By,Bz,L,dz):
    z=0
    #n_steps=int((L-z)/dz)
    n_steps=int(L*np.log(100)/dz)
    M=np.matrix(np.identity(3))
    R1=np.matrix(np.identity(3))
    R2=np.matrix(np.identity(3))
    
    
    B=np.sqrt(Bx**2+By**2+Bz**2)
    theta=np.arccos(Bz/B)
    omega_c=e*B/(m*c)
    
    ne_arr=[]
    psi2_arr=[]
    chi2_arr=[]
    mu1_arr=[]
    mu2_arr=[]
    temp=0
    
    for i in range(n_steps):
        #ne=(np.exp(z/L)-1)*nc/(e_eular-1)
        ne=nc*np.exp(z/L-np.log(100))
        omega_p=np.sqrt(4*np.pi*ne*e**2/m)
        # omega_c=e*B/(m*c)
        
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
        
        if (mu2>0):
            prefactor=omega_p**2/((mu1+mu2)*c*omega**3*D)
            
            O1=prefactor*(e/(m*c))**2*((Bx**2-By**2)/(1-omega_p**2/omega**2))
            
            O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-omega_p**2/omega**2))
            
            O3=prefactor*2*omega*e*Bz/(m*c)
            
            O=np.sqrt(O1**2+O2**2+O3**2)
            
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
            
            
            s=np.array(np.dot(M,s0))
            s=s.reshape((3,1))
            #print(s)
            try:
                res=np.arctan(s[1][0]/s[0][0])
                if(np.isnan(res)):
                    psi2_arr.append(np.pi/2)
                else:
                    psi2_arr.append(res)
            except:
                None
            try:
                res=np.arctan(s[2][0]/np.sqrt((s[0][0])**2+s[1][0])**2)
                if(np.isnan(res)):
                    chi2_arr.append(np.pi/2)
                else:
                    chi2_arr.append(res)
            except:
                None
            
            
        elif(mu2==0):
            if(temp==0):
                s1=np.dot(M,s0)
                A1 = s1[1]/s1[0]      #tan2psi
                B1 = s1[2]           #sin2chi
                
                s2eta = np.sqrt((A1**2+B1**2)/(1+A1**2))
                t2eta = s2eta/np.sqrt(1-s2eta**2)
                phi = np.arcsin(B1/s2eta)
                if(np.isnan(phi)):
                    phi = 0
                temp=1
                
            phi = -(mu1*dz*omega/c)+phi
            
            psiN=np.arctan(t2eta*np.cos(phi))            
            #print(psiN)
            chiN=np.arcsin(s2eta*np.sin(phi))
                                                                         
            psi2_arr.append(psiN[0,0])
            chi2_arr.append(chiN[0,0])
            
        z+=dz
        
        mu1_arr.append(mu1)
        mu2_arr.append(mu2)
        ne_arr.append(ne)
        
    s2_1=[np.cos(chi2_arr[-1])*np.cos(psi2_arr[-1])]
    s2_2=[np.cos(chi2_arr[-1])*np.sin(psi2_arr[-1])]
    s2_3=[np.sin(chi2_arr[-1])]

    s2=np.array([s2_1,s2_2,s2_3])
    s2=s2.reshape((3,1))

    M=np.matrix(np.identity(3))
    temp=0
    
    for i in range(n_steps):
        #ne=(np.exp(z/L)-1)*nc/(e_eular-1)
        ne=nc*np.exp(z/L-np.log(100))
        omega_p=np.sqrt(4*np.pi*ne*e**2/m)
        #omega_c=e*B/(m*c)
        
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
        
        if (mu2>0):
            if(temp==0):
                s3_1=np.cos(chi2_arr[-1])*np.cos(psi2_arr[-1])
                s3_2=np.cos(chi2_arr[-1])*np.sin(psi2_arr[-1])
                s3_3=np.sin(chi2_arr[-1])
                
                s3=np.array([s3_1,s3_2,s3_3])
                s3=s3.reshape((3,1))
            
                temp=1
                
            prefactor=omega_p**2/((mu1+mu2)*c*omega**3*D)
            
            O1=prefactor*(e/(m*c))**2*((Bx**2-By**2)/(1-omega_p**2/omega**2))
            
            O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-omega_p**2/omega**2))
            
            O3=prefactor*2*omega*e*Bz/(m*c)
            
            O=np.sqrt(O1**2+O2**2+O3**2)
            
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
            
            s=np.dot(M,s3)
            s=s.reshape((3,1))
            #print(s)
            try:
                res=np.arctan(s[1][0]/s[0][0])
                if(np.isnan(res)):
                    psi2_arr.append(np.pi/2)
                else:
                    psi2_arr.append(res[0,0])

            except:
                None
            try:
                res=np.arctan(s[2][0]/np.sqrt((s[0][0])**2+s[1][0])**2)
                if(np.isnan(res)):
                    chi2_arr.append(np.pi/2)
                else:
                    chi2_arr.append(res[0,0])
            except:
                None

        
        elif(mu2==0):
            try:
                phi = -(mu1*dz*omega/c)+phi
            except:
                s1=np.dot(M,s0)
                A1 = s1[1]/s1[0]      #tan2psi
                B1 = s1[2]           #sin2chi
                
                s2eta = np.sqrt((A1**2+B1**2)/(1+A1**2))
                t2eta = (s2eta/np.sqrt(1-s2eta**2))
                phi = np.arcsin(B1/s2eta)
                if (np.isnan(phi)):
                    phi=0
                
            psiN=np.arctan(t2eta*np.cos(phi))            
            chiN=np.arcsin(s2eta*np.sin(phi))
            
            psi2_arr.append(psiN[0,0])
            chi2_arr.append(chiN[0,0])
            
        z-=dz
        
        mu1_arr.append(mu1)
        mu2_arr.append(mu2)
        ne_arr.append(ne)
        
    
        
    sf=np.array(np.dot(M,s3))
    
    plt.plot(np.linspace(0,2*L*np.log(100),len(ne_arr))/(L*np.log(100)),np.array(ne_arr)/nc,'r-')
    plt.title("variation of ne \n"+"units in cgs")
    plt.xlabel("z/(L ln 100)")
    plt.ylabel("ne")
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()
    
    
    #print(psi2_arr)
    psi2_arr=phase_unwrap(psi2_arr)
    chi2_arr=phase_unwrap(chi2_arr)
    plt.plot(np.linspace(0,2*L*np.log(100),len(psi2_arr))/(L*np.log(100)),psi2_arr/2,'g-',label=r"$\psi$")
    plt.plot(np.linspace(0,2*L*np.log(100),len(chi2_arr))/(L*np.log(100)),chi2_arr/2,'r-',label=r"$\chi$")
    plt.title(r"variation of $\psi$ and $\chi$"+"\nunit in rad")
    plt.xlabel("z/(L ln 100)")
    plt.ylabel("psi")
    plt.legend()
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()
    
    plt.plot(np.linspace(0,2*L*np.log(100),len(mu1_arr))/(L*np.log(100)),mu1_arr,'r-',label=r"$\mu_1$")
    plt.plot(np.linspace(0,2*L*np.log(100),len(mu2_arr))/(L*np.log(100)),mu2_arr,'b-',label=r"$\mu_2$")
    plt.legend()
    plt.title(r"variation of $\mu$")
    plt.xlabel("z/(L ln 100)")
    plt.ylabel(r"$\mu$")
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()
    
    
    return(sf)

#final_stokes=np.vectorize(final_stokes)


chi=0
psi=0.0


Bx = 80e6
By = 80e6
Bz = 10

t=1
L=float(t*1e-12*6e6)
dz=1e-8

s0_1=[np.cos(2*chi)*np.cos(2*psi)]
s0_2=[np.cos(2*chi)*np.sin(2*psi)]
s0_3=[np.sin(2*chi)]

s0=np.array([s0_1,s0_2,s0_3])

sf=final_stokes(s0,Bx,By,Bz,L,dz)
print("s0: \n",s0)
print("sf: \n",sf)
print("sf-s0: \n",sf-s0)

final_chi=np.arctan(sf[2][0]/np.sqrt((sf[0][0])**2+sf[1][0])**2)/2
print(f"final chi: {final_chi}")

final_psi=np.arctan(sf[1][0]/sf[0][0])/2
print(f"final psi:  {final_psi}")

print(f"Time:  {time.time()-t_start}")