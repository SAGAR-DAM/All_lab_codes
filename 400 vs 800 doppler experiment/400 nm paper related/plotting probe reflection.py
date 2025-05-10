import numpy as np
from mayavi import mlab

# Create a 3D grid: more resolution in x, fewer in y and z
x, y, z = np.mgrid[0:5:200j, 0:5:20j, 0:5:20j]




def density(x, y, z, a):
    return np.exp(-(x - 2)) + a*np.sin(15 * x) / (1 + x**2)

f = density(x, y, z, 0)

linex = np.linspace(0,7,200)
liney = np.ones(len(linex))
linez = np.ones(len(linex))*2.5
linez = density(linex,liney,linez,0.5)

# Create square figure window
mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))

# Plot isosurface
src = mlab.contour3d(f, contours=200, opacity=0.02,colormap="plasma")
# Draw the line (black color, tube style)
line = mlab.plot3d(linex*40,liney*10,linez*20/np.max(linez), color=(0, 0, 0), tube_radius=0.15)
line.actor.actor.scale = [1, 10, 10]

liney = np.linspace(-1,6,100)
linez = np.ones(len(liney))
linex = 20*(liney-2.5)**2

line1 = mlab.plot3d(linex+20,liney+7.5,linez*12, color=(0, 0, 1), tube_radius=0.5)
line1.actor.actor.scale = [1, 10, 10]

n=len(linex)
line3 = mlab.plot3d([(linex+20)[n//2],(linex+20)[n//2]],[(liney+7.5)[n//2],(liney+7.5)[n//2]],[(linez*12)[n//2],0], color=(0, 0, 0), tube_radius=0.2)
line3.actor.actor.scale = [1, 10, 10]
line4 = mlab.plot3d([(linex+20)[n//2],(linex+20)[-12]],[(liney+7.5)[n//2],(liney+7.5)[n//2]],[0,0], color=(0, 0, 0), tube_radius=0.2)
line4.actor.actor.scale = [1, 10, 10]
# arrow1 = mlab.quiver3d((linex+120)[0],10*(liney+7.5)[0],10*(linez*12)[0],10*np.diff(linex+20)[0],10*np.diff(liney+7.5)[0],10*np.diff(linez*12)[0], color=(0, 0, 1), mode='cone', scale_factor=1)
arrow1 = mlab.quiver3d((linex)[-1],10*(liney+7.5)[-1],10*(linez*12)[-1],10*np.diff(linex+20)[-1],10*np.diff(liney+7.5)[-1],10*np.diff(linez*12)[-1], color=(0, 0, 1), mode='cone', scale_factor=0.7)
# arrow1.actor.actor.scale = [1, 10, 10]
# Rescale the actor to make it appear cubic
src.actor.actor.scale = [1, 10, 10]  # scale Y and Z axes to match X
# Optional: axes and title
# mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
# mlab.title('Contour3D: Isosurfaces of f(x, y, z)')
mlab.show()
