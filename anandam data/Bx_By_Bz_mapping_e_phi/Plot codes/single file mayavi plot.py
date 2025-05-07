# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:49:58 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display


filename = "D:\\data Lab\\anandam data\\Bx_By_Bz_mapping_e_phi\\t_5.33\\ellipticity\\Bx_By_to_ellipticity_t_5.33_Bz_1.0.txt"

f = open(filename,"r")
r = np.loadtxt(f)
r = np.flip(r,axis=0)

plt.imshow(r, cmap="jet", extent = [-100,100,-100,100])
plt.title(r"ellipticity vs $\vec B$"+"\n"+filename[88:-4])
plt.xlabel("Bx  (MG)")
plt.ylabel("By  (MG)")
plt.colorbar()
plt.show()

Bx, By = np.mgrid[-100:100:201j, -100:100:201j]

lensoffset=0
xx = yy = zz = np.arange(-100,100,1)
xy = xz = yx = yz = zx = zy = np.zeros_like(xx)    
mlab.plot3d(yx,yy+lensoffset,yz,line_width=0.01,tube_radius=0.1)
mlab.plot3d(zx,zy+lensoffset,zz,line_width=0.01,tube_radius=0.1)
mlab.plot3d(xx,xy+lensoffset,xz,line_width=0.01,tube_radius=0.1)
mlab.mesh(Bx,By,200*r,representation='surface')
mlab.axes(extent=[-100, 100, -100, 100, 200*np.min(r), 200*np.max(r)], color=(0, 0, 0), nb_labels=5)
mlab.xlabel("Bx")
mlab.ylabel("By")
mlab.zlabel("ellipticity \n(200X)")