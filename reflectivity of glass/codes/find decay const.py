import numpy as np
import matplotlib.pyplot as plt
import glob
from Curve_fitting_with_scipy import exp_decay_fit as eft
import pandas as pd 

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

c = 0.3
def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index

def zero_indexing(x,y):
    y /= max(y)
    xmax_index =find_index(y,max(y))
    xmax_val = x[xmax_index]
    
    x -= xmax_val
    return(x)
    
df = pd.read_csv("D:\\data Lab\\reflectivity of glass\\800 nm pump\\run 22 glass\\MeasLog.csv")
data = df['CH2 - PK2Pk']
data -= np.mean(data[-10:])


delay = np.linspace(22.5,26,len(data))
delay = zero_indexing(delay, data)
delay = 2*delay/c

minw = find_index(data, max(data))

eff_delay = delay[minw:]
eff_data = data[minw:]

fit_y, parameters = eft.fit_exp_decay(eff_delay, eff_data)
print(parameters)

plt.plot(delay, data,'ro--',markersize=4, lw=0.5)
plt.plot(eff_delay, fit_y, 'r-',lw=3, label = fr'800 pump, $\tau$: {parameters[1]:.2f} ps')
plt.plot(eff_delay, fit_y, 'y-', lw=0.5)





df = pd.read_csv("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14th Feb 2024\\pd signal\\Wed Feb 14 16_24_24 2024\\MeasLog.csv")
data = df['CH2 - PK2Pk']
data -= np.mean(data[-10:])

delay = np.linspace(9.5,13.5,len(data))
delay = zero_indexing(delay, data)
delay = 2*delay/c

minw = find_index(data, max(data))

eff_delay = delay[minw:]
eff_data = data[minw:]

fit_y, parameters = eft.fit_exp_decay(eff_delay, eff_data)
print(parameters)

plt.plot(delay, data,'bo--',lw= 0.5, markersize=4)
plt.plot(eff_delay, fit_y, 'b-', lw=3, label = fr'400 pump, $\tau$: {parameters[1]:.2f} ps')
plt.plot(eff_delay, fit_y, 'y-', lw=0.5)


plt.legend()
plt.xlabel("delay (ps)")
plt.ylabel("Normalized Reflected signal (arb unit)")
plt.title("Decay constant modelled as: "+r"$y\propto e^{-t/\tau}$")
plt.xlim(-5,max(delay))
plt.grid(color='k',lw=0.3)
plt.minorticks_on()
plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=10)
plt.show()
