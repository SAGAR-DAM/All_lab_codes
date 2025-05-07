# -*- coding: utf-8 -*-
"""
Created on Sat May  3 14:36:42 2025

@author: mrsag
"""
import numpy as np

def Te(I,l,tau):
    temp = (I*l**2)**0.55*tau**0.27
    return temp

def collission(Z,T):
    coeff = 2.9e-6
    coloumb = 10
    ne = 6.97e21
    
    val = coeff*coloumb*ne*Z/T**1.5
    
    return val

def vel(nu,tau):
    c = 3e8
    val = 3*c/(8*nu*tau)
    return val

factor = Te(2e18,800,30)/Te(3e17,400,200)
T = 134 # 400/factor
Z = 5  # 4.63 from simulation
tau = 2.02e-12

nu = collission(Z, T)
v = vel(nu,tau)*1e2

print(f"v: {v:.2e} cm/s")

