# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:39:15 2023

@author: sagar

find reference in: 
    
https://pdf.sciencedirectassets.com/271557/1-s2.0-S0030401807X07983/1-s2.0-S003040180700555X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBQaCXVzLWVhc3QtMSJHMEUCIQDZWzs%2Bdqmiv0xwGWkCY0t68A%2BXWB5bWwLYXvyvTJOJrgIgOSax7xgKA%2F88yRRfR%2BK6v%2FWi35S%2BxrKr6mnEfOzywBAqsgUIXRAFGgwwNTkwMDM1NDY4NjUiDLabc%2F%2B%2B3bk1JUid%2BiqPBTnTsFfBMixZcbDz2%2BsJQjDQD%2FQr7AykmmnOCv1k%2BCreZ3PBagyTeryb1KacpfqoqZ6DheUVGeV103fGtOiPJdAJffdUVQ%2F9EAGCrAk0Mo%2BQDWMeuxiajaP66wkcptSZUBEmpbSHRh89WBv962Ys7BMs57nCFZvbgq4Md6LA1CTL3smz4M1k%2F3NtyjKO3FJp%2FF1tkF1960D%2F4mdXLaZNd5NaqpjWCimUn9ZOY4gXOJmk6ZQk7BdXMpPlzPDnuWhXBSx4R1unhxltEw4DYaTAeNf2vaiHZEwSUx3Wfi%2B4%2FcGEtOh4CVCSJECXVHfs7PrMyNLQlUMIJC5RHmArFvpy%2Br8xjJd%2Fpbi0tybcfC2zCrDD138xO%2FqUmQa9mcm%2Ffp6p6eQBwanFKf2j97pS10X%2FSSL9%2FBB9X3IXRBAoMjuz3KdDCUb7Fpi0iXhayFzb0ko25VL6uitew6PAlE7kb3XHcf2SNIAyVyMrYu0fQ1vbT2QeuhEVXr2GHn2eL590xs28NIOxpJ3UbSa%2FB2APYaPqa7fau7VzTyp%2B1I3l54B548GtLg4jssVsqk%2FzAngXIQuks%2FcYA%2BOjQOpnw1eLimNoV%2BKrLZZOyJ7Hx01ZywrR39CXGPAtkohQ2ddBPgD2p0un%2By9aSbwPQAkMLtaFNSx%2Bupiq1hAA12iEKlx4tlKrLM0IFQFiXAT%2Bser1nvUlYylB3774Nqe41igvkaCtZ0M9MHNlDbf9j%2FJ85USX4T0R325PvXG2TS2jK%2BwTSDYf2PuLHalzrE7W%2BNTDhoJ89%2FGYJfEfI1twNQusRXfhvMcg8TL1zoz4KA29wFjxxbZ7KPrzgM%2FR9R3TuZLBzCsWYD8nbcT50v6QXp3uDQA9PIZ3%2Fqsw%2FtOwvgY6sQEbv0D%2BVrQjqEJBuRYnwFvjo9NdsbaMkUhbyDTE0cOhw4eCwEZI5lZwOCyH5M%2FZ5jax1tfFMmjT97RejPV4cPnV7nxFJjLnGgNi%2FVWdCLjIW2WMJu2pcTFacqu1YS3Ja3RiP9rQ3V1EG0A7A2St3YpEIIA52mBnA3VxBkVYL9bs%2FfZpWFcv8sIsCotwmXDKHOwPx7%2B%2B85DZG2ozf5b8FoR%2FbhuMIENPwe%2BJf%2B5oDuRVTIw%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250308T122015Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY3ES4PZP6%2F20250308%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=15da62efde747ee66d715fd521b5c93c8fc1fc207848d18269d57e2ee19c5b6f&hash=1cef61ff217f3e35cd17b6e2de72d8b492e3922e220c12d2049a428929e2d25b&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S003040180700555X&tid=spdf-9ec96685-17c2-4fd5-9542-cb03e8026d1d&sid=a36784b16546874b0a7ad8f98851224919edgxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f0d5b510150505a5001&rr=91d2491d1ab38af9&cc=in
https://www.edmundoptics.in/knowledge-center/application-notes/lasers/gaussian-beam-propagation/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import glob
from Curve_fitting_with_scipy import Linefitting as lft

import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

files=glob.glob("D:\\data Lab\\zscan april 2023\\4 photon absorption\\*.csv")

#Crystal information
L=100*10**(-6)   # sample thickness
w0=125*10**(-6)    # beam waist at focus
E=np.array([20.3,12,52.8])*10**(-6)    # incident energies
E_err=np.array([0.912,0.74,2.52])*10**(-6)
dt=45*10**(-15)    #   pulse width (here 30 fs)
I=E/(dt*np.pi*w0**2)      # 52.8: 1.436*10**17     20.3: 5.523*10**16       12: 3.265*10**16
I_err=E_err/(dt*np.pi*w0**2)*10**(-16)
n0=1.5
lamda=800*10**(-9)    # wavelength

#zR=np.pi*w0**2*n0/lamda
zR_array=np.array([3.74,2.14,4.2])*10**(-3)    # 52.8: 1.14         20.3: 1.14      12: 2.14



def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index


def moving_avg(arr,n):
    window_size = n
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
    
    # Loop through the array t o
    #consider every window of size 3
    while i < len(arr) - window_size + 1:
    
    	# Calculate the average of current window
    	window_average = round(np.sum(arr[
    	i:i+window_size]) / window_size, 2)
    	
    	# Store the average of current
    	# window in moving average list
    	moving_averages.append(window_average)
    	
    	# Shift window to right by one position
    	i += 1
    return(moving_averages)


def point_avg(arr,n):
    arr1=[]
    for i in range(int(len(arr)/n)):
        x=np.mean(arr[n*i:n*(i+1)])
        arr1.append(x)
    arr1.append(np.mean(arr[(int(len(arr)/n))*n:]))
    
    return(arr1)

filenumber = 2
file = files[filenumber]

data=pd.read_csv(file)
print(data)

I0=I[filenumber]
zR=zR_array[filenumber]
x=data["CH1 - PK2Pk"]
#x=x*3000/max(x)
y=data["CH4 - PK2Pk"]

x=x/y*y[0]
x=x/max(x)
x=x/np.mean(x[0:100])
x=point_avg(x,5)
print(len(x))

position=np.linspace(0,30,len(x))
xmin_index=find_index(x, min(x))

position=position-xmin_index/len(x)*max(position)
minval=0

position=position[minval:]-(position[minval]-position[0])
x=x[minval:]

pos_in_meter = position*1e-3


plt.plot(position,x)
plt.show()



def z_to_I(z):
    # w = w0*np.sqrt(1+((lamda*z)/(np.pi*w0**2))**2)
    w = w0*np.sqrt(1+(z/zR_array[filenumber])**2)
    intensity = E[filenumber]/(dt*np.pi*w**2)
    return intensity

I_val = z_to_I(pos_in_meter)
plt.plot(position,I_val)
plt.xlabel("distance from focus (mm)")
plt.ylabel("Intenisity   "+r"($W/m^2$)")
plt.show()


pos_index_at_absorption_beginning = find_index(position,-2.95)
pos_index_at_0 = find_index(position,0)

I_val_downslope = I_val[pos_index_at_absorption_beginning:pos_index_at_0]
x_downslope = x[pos_index_at_absorption_beginning:pos_index_at_0]
absorption = 1-np.array(x_downslope)

fit_log_absorption, parameters = lft.linefit(np.log(I_val_downslope),np.log(absorption))
slope = parameters[0]

plt.plot(np.log(I_val_downslope),np.log(absorption),"ro-",lw=0.5)
plt.plot(np.log(I_val_downslope),fit_log_absorption,"k-")
plt.xlabel("log(I)  (dimensionless)")
plt.ylabel("log(Absorption)")
plt.grid(color="k",lw=0.5)
plt.title("log-log plot of absorption vs normalized absorpition"+"\n"+r"Peak input intensity: %.2f"%(I[filenumber]*10**(-16))+r"$\pm$%.2f"%(I_err[filenumber])+r" $TW/cm^2$"+f"\nfitted slope: {slope:.2f}")
plt.show()


