# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:08:06 2023

@author: sagar
"""

import numpy as np


#defining all constants in CGS
e=4.8032e-10                        #electron charge
m=9.1094e-28                        #electron mass
c=2.99792458e10                     #speed of light
wavelength=4e-5                     #used probe wavelength
omega=2*np.pi*c/wavelength          #prob angular frequency
nc=omega**2*m/(4*np.pi*e**2)
e_eular=2.718281

class state:
    def __init__(self,Bx,By,Bz,L,z,dz):
        self.Bx=Bx
        self.By=By
        self.Bz=Bz
        self.z=z
        self.dz=dz
        self.L=L
        
        state.B=np.sqrt(self.Bx**2+self.By**2+self.Bz**2)
        state.theta=np.arccos(self.Bz/np.sqrt(self.Bx**2+self.By**2+self.Bz**2))
        state.beta=np.pi/2-np.arctan(self.By/self.Bz)
        
        state.ne=e_eular*nc/(e_eular-1)*(1-np.exp(-self.z/self.L))
        state.omega_p=np.sqrt(4*np.pi*state.ne*e**2/m)
        state.omega_c=e*state.B/(m*c)
        
        state.F=2*omega/state.omega_c*(1-state.omega_p**2/omega**2)*np.cos(state.theta)/(np.sin(state.theta))**2
        state.mu1_sq=1-state.omega_p**2/omega**2*1/(1+state.omega_c**2/omega**2*(np.sin(state.theta))**2/(2*(1-state.omega_p**2/state.omega_c**2))*(-1+np.sqrt(1+state.F**2)))
        state.mu2_sq=1-state.omega_p**2/omega**2*1/(1+state.omega_c**2/omega**2*(np.sin(state.theta))**2/(2*(1-state.omega_p**2/state.omega_c**2))*(-1-np.sqrt(1+state.F**2)))
        
        state.mu1=np.sqrt(state.mu1_sq)
        state.mu2=np.sqrt(state.mu2_sq)
        
        state.N=(state.omega_p/omega)**2
        state.D=1-(e/(m*state.omega_c))**2*((self.Bx**2+self.By**2)/(1-state.N)+self.Bz**2)
        
    def make_Omega(self):
        if np.isnan(state.mu1):
            state.mu1=1
        if np.isnan(state.mu2):
            state.mu2=1
            
        prefactor=state.omega_p**2/((state.mu1+state.mu2)*c*omega**3*state.D)
        
        O1=prefactor*(e/(m*c))**2*((self.Bx**2-self.By**2)/(1-state.omega_p**2/omega**2))
        
        O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-state.omega_p**2/omega**2))
        
        O3=prefactor*2*omega*e*self.Bz/(m*c)
        
        return np.array([O1,O2,O3])
        
        
def final_stocks(s0,Bx,By,Bz,L,dz):
    n_steps=int(L/dz)
    z=0
    M=np.matrix(np.identity(3))
    I=np.matrix(np.identity(3))
    A=np.matrix(np.identity(3))
    
    current_state=state(Bx=Bx, By=By, Bz=Bz, L=L, z=z, dz=dz)
    
    for i in range(n_steps):
        O1=current_state.make_Omega()[0]
        O2=current_state.make_Omega()[1]
        O3=current_state.make_Omega()[2]
        
        O=np.sqrt(O1**2+O2**2+O3**2)
        
        A[0,1]=-O3
        A[0,2]=O2
        
        A[1,0]=O3
        A[1,2]=-O1
        
        A[2,0]=-O2
        A[2,1]=O1
        
        M=(I  +  np.sin(O*dz)/O*A  +  1/2*(np.sin(O*dz)/O)**2*A**2)*M
        
        z+=dz
        current_state.z=z
        print(current_state.mu2)
    sf=np.dot(M,s0)
    return(sf)

chi=0.1
psi=0.01

Bx=100e6
By=120e6
Bz=50e6

L=3e-12*6e6
dz=L/100

s0_1=[np.cos(2*chi)*np.cos(2*psi)]
s0_2=[np.cos(2*chi)*np.sin(2*psi)]
s0_3=[np.sin(2*chi)]

s0=np.array([s0_1,s0_2,s0_3])

sf=final_stocks(s0,Bx,By,Bz,L,dz)
print(sf)