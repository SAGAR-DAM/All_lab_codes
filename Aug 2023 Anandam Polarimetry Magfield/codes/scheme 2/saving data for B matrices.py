import numpy as np

# Your matrix (replace this with your actual matrix)
r,phi=np.mgrid[0.00001:100.00001:101j,0:2*np.pi:101j]
Bx=r*np.cos(phi)
By=r*np.sin(phi)
#X,Y = np.mgrid[0:100:200j,0:100:200j]
print(Bx)
#print(Bx[156][56])


file_name = "D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\scheme 2\\matrices\\B matrices\\Bx.txt"

# Save the matrix to a text file
np.savetxt(file_name, Bx, fmt='%f', delimiter='\t')


file_name = "D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\scheme 2\\matrices\\B matrices\\By.txt"

# Save the matrix to a text file
np.savetxt(file_name, By, fmt='%f', delimiter='\t')

e_0=np.zeros(shape=Bx.shape)
print(Bx.shape)
print(e_0.shape)
# Specify the file name
file_name = "D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\scheme 2\\matrices\\e matrices\\e0000_t_0_0.txt"

# Save the matrix to a text file
np.savetxt(file_name, e_0, fmt='%f', delimiter='\t')

# Optional: Display a message
print(f"Matrix saved to {file_name}")

#file_name = "matrix.txt"

# Load the matrix from the text file
r = np.loadtxt(file_name, dtype=float, delimiter='\t')

f=r[:][:]
print(np.max(f-Bx))
