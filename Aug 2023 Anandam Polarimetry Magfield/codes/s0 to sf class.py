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
nc=omega**2*m/(4*np.pi*e**2)        #plasma critical frequency
e_eular=2.718281                    #eular's const

class state:                        #defining the state corresponding to a particular z
    def __init__(self,Bx,By,Bz,L,z,dz):         #initiliaze the class
        self.Bx=Bx          #from the user
        self.By=By          #from the user
        self.Bz=Bz          #from the user
        self.z=z            #from the user
        self.dz=dz          #from the user
        self.L=L            #from the user
        
        # Defining the auxuliary variables
        state.B=np.sqrt(self.Bx**2+self.By**2+self.Bz**2)
        state.theta=np.arccos(self.Bz/state.B)
        state.beta=np.pi/2-np.arctan(self.By/self.Bz)
        
        state.ne=(np.exp(self.z/self.L)-1)*nc/(e_eular-1)       #Electron density function, Exponential model
        state.omega_p=np.sqrt(4*np.pi*state.ne*e**2/m)          #plasma frequency
        state.omega_c=e*state.B/(m*c)                           #cyclotron freq
        
        state.F=2*omega/state.omega_c*(1-state.omega_p**2/omega**2)*np.cos(state.theta)/(np.sin(state.theta))**2
        state.mu1_sq=1-state.omega_p**2/omega**2*1/(1+state.omega_c**2/omega**2*(np.sin(state.theta))**2/(2*(1-state.omega_p**2/state.omega_c**2))*(-1+np.sqrt(1+state.F**2)))          #Ordinary refractive index sq
        state.mu2_sq=1-state.omega_p**2/omega**2*1/(1+state.omega_c**2/omega**2*(np.sin(state.theta))**2/(2*(1-state.omega_p**2/state.omega_c**2))*(-1-np.sqrt(1+state.F**2)))          #Extraordinary refractive index sq
        
        state.mu1=np.sqrt(state.mu1_sq)     #ord refractive index
        state.mu2=np.sqrt(state.mu2_sq)     #extraord refractive index
        
        state.N=(state.omega_p/omega)**2
        state.D=1-(e/(m*omega*c))**2*((self.Bx**2+self.By**2)/(1-state.N)+self.Bz**2)
        
    def make_Omega(self):                   #Making the Omega matrix
        if np.isnan(state.mu1):
            state.mu1=0
        if np.isnan(state.mu2):
            state.mu2=0
            
        prefactor=state.omega_p**2/((state.mu1+state.mu2)*c*omega**3*state.D)                   #prefactor for all components
        
        O1=prefactor*(e/(m*c))**2*((self.Bx**2-self.By**2)/(1-state.omega_p**2/omega**2))       #Omega_1
        
        O2=prefactor*(e/(m*c))**2*(2*self.Bx*self.By/(1-state.omega_p**2/omega**2))                       #Omega_2
        
        O3=prefactor*2*omega*e*self.Bz/(m*c)                                                    #Omega_3
    
        return np.array([O1,O2,O3])
    
    def update_values(self):                    #Update the state variables as the loop goes through
        state.ne=(np.exp(self.z/self.L)-1)*nc/(e_eular-1)
        state.omega_p=np.sqrt(4*np.pi*state.ne*e**2/m)
        state.omega_c=e*state.B/(m*c)
        
        state.F=2*omega/state.omega_c*(1-state.omega_p**2/omega**2)*np.cos(state.theta)/(np.sin(state.theta))**2
        state.mu1_sq=1-state.omega_p**2/omega**2*1/(1+state.omega_c**2/omega**2*(np.sin(state.theta))**2/(2*(1-state.omega_p**2/omega**2))*(-1+np.sqrt(1+state.F**2)))
        state.mu2_sq=1-state.omega_p**2/omega**2*1/(1+state.omega_c**2/omega**2*(np.sin(state.theta))**2/(2*(1-state.omega_p**2/omega**2))*(-1-np.sqrt(1+state.F**2)))
        
        state.mu1=np.sqrt(state.mu1_sq)
        state.mu2=np.sqrt(state.mu2_sq)

        state.N=(state.omega_p/omega)**2
        state.D=1-(e/(m*omega*c))**2*((self.Bx**2+self.By**2)/(1-state.N)+self.Bz**2)
        
        
def phase_unwrap(arr):          #function to unwrap the phase for psi and chi
    arr=np.array(arr)
    delta=np.max(arr)-np.min(arr)
    n=int(2*np.pi/delta)
    arr=arr*n
    arr=np.unwrap(arr)
    arr/=n
    return(arr)
        
def final_stokes(s0,Bx,By,Bz,L,dz):
    n_steps=int(L/dz)
    z=0
    M=np.matrix(np.identity(3))
    I=np.matrix(np.identity(3))
    A=np.matrix(np.identity(3))
    
    ne_arr=[]
    psi2_arr=[]
    chi2_arr=[]
    mu1_arr=[]
    mu2_arr=[]
    
    current_state=state(Bx=Bx, By=By, Bz=Bz, L=L, z=z, dz=dz)
    
    for i in range(n_steps):
        O1=current_state.make_Omega()[0]
        O2=current_state.make_Omega()[1]
        O3=current_state.make_Omega()[2]

        O=np.sqrt(O1**2+O2**2+O3**2)
        
        A[0,0]=0
        A[0,1]=-O3
        A[0,2]=O2
        
        A[1,0]=O3
        A[1,1]=0
        A[1,2]=-O1
        
        A[2,0]=-O2
        A[2,1]=O1
        A[2,2]=0
        
        if(O==0):
            M=(I  +  dz*A  +  1/2*dz**2*A**2)*M
        else:
            M=(I  +  np.sin(O*dz)/O*A  +  1/2*(np.sin(O*dz)/O)**2*A**2)*M
        
        sf=np.array(np.dot(M,s0))
        ne_arr.append(current_state.ne)
        try:
            val=np.arctan(sf[1][0]/sf[0][0])
            if(np.isnan(val)):
                psi2_arr.append(np.pi/2)
            else:
                psi2_arr.append(val)
        except:
            None
        try:
            val=np.arctan(sf[2][0]/np.sqrt((sf[0][0])**2+sf[1][0])**2)
            if(np.isnan(val)):
                chi2_arr.append(np.pi/2)
            else:
                chi2_arr.append(val)
        except:
            None
        mu1_arr.append(current_state.mu1)
        mu2_arr.append(current_state.mu2)
        
        z+=dz
        
        current_state.z=z
        current_state.update_values()
        
    for i in range(n_steps):
        O1=current_state.make_Omega()[0]
        O2=current_state.make_Omega()[1]
        O3=current_state.make_Omega()[2]

        O=np.sqrt(O1**2+O2**2+O3**2)
        
        A[0,0]=0
        A[0,1]=-O3
        A[0,2]=O2
        
        A[1,0]=O3
        A[1,1]=0
        A[1,2]=-O1
        
        A[2,0]=-O2
        A[2,1]=O1
        A[2,2]=0
        
        if(O==0):
            M=(I  +  dz*A  +  1/2*dz**2*A**2)*M
        else:
            M=(I  +  np.sin(O*dz)/O*A  +  1/2*(np.sin(O*dz)/O)**2*A**2)*M
        
        sf=np.array(np.dot(M,s0))
        ne_arr.append(current_state.ne)
        try:
            val=np.arctan(sf[1][0]/sf[0][0])
            if(np.isnan(val)):
                psi2_arr.append(np.pi/2)
            else:
                psi2_arr.append(val)
        except:
            None
        try:
            val=np.arctan(sf[2][0]/np.sqrt((sf[0][0])**2+sf[1][0])**2)
            if(np.isnan(val)):
                chi2_arr.append(np.pi/2)
            else:
                chi2_arr.append(val)
        except:
            None
        mu1_arr.append(current_state.mu1)
        mu2_arr.append(current_state.mu2)
        
        z-=dz
        
        current_state.z=z
        current_state.update_values()

    sf=np.array(np.dot(M,s0))
    
    plt.plot(np.arange(0,2*L,dz)/L,np.array(ne_arr)/nc,'r-')
    plt.title("variation of ne \n"+"units in cgs")
    plt.xlabel("z/L")
    plt.ylabel("ne")
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()
    
    
    psi2_arr=phase_unwrap(psi2_arr)
    chi2_arr=phase_unwrap(chi2_arr)
    plt.plot(np.linspace(0,2*L,len(psi2_arr))/L,psi2_arr/2,'g-',label=r"$\psi$")
    plt.plot(np.linspace(0,2*L,len(chi2_arr))/L,chi2_arr/2,'r-',label=r"$\chi$")
    plt.title(r"variation of $\psi$ and $\chi$"+"\nunit in rad")
    plt.xlabel("z/L")
    plt.ylabel("psi")
    plt.legend()
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()
    
    plt.plot(np.linspace(0,2*L,len(mu1_arr))/L,mu1_arr,'r-',label=r"$\mu_1$")
    plt.plot(np.linspace(0,2*L,len(mu2_arr))/L,mu2_arr,'b-',label=r"$\mu_2$")
    plt.legend()
    plt.title(r"variation of $\mu$")
    plt.xlabel("z/L")
    plt.ylabel(r"$\mu$")
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()
    
    
    return(sf)

chi=0.0     #initial chi
psi=0.0     #initial psi

Bx=100e6    #initial B
By=100e6
Bz=100e6

L=5e-12*6e6 #Plasma total length
dz=L/200   #one slab length

s0_1=[np.cos(2*chi)*np.cos(2*psi)]
s0_2=[np.cos(2*chi)*np.sin(2*psi)]
s0_3=[np.sin(2*chi)]

s0=np.array([s0_1,s0_2,s0_3])

sf=final_stokes(s0,Bx,By,Bz,L,dz)
print("s0: \n",s0)
print("sf: \n",sf)
print("sf-s0: \n",sf-s0)


for __var__ in dir():
    exec('del '+ __var__)
    del __var__
    
import sys
sys.exit()