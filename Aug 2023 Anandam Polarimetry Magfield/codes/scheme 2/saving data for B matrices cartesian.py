import numpy as np

# Your matrix (replace this with your actual matrix)
Bx,By=np.mgrid[0.001:100.001:51j,0.001:100.001:51j]

Bx=np.concatenate([Bx, -Bx, -Bx, Bx])
By=np.concatenate([By, By, -By, -By])


file_name = "D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\scheme 2\\Fast algo matrices\\B matrices cartesian\\Bx.txt"

# Save the matrix to a text file
np.savetxt(file_name, Bx, fmt='%f', delimiter='\t')


file_name = "D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\scheme 2\\Fast algo matrices\\B matrices cartesian\\By.txt"

# Save the matrix to a text file
np.savetxt(file_name, By, fmt='%f', delimiter='\t')

e_0=np.zeros(shape=Bx.shape)
print(Bx.shape)
print(e_0.shape)
# Specify the file name
file_name = "D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\scheme 2\\Fast algo matrices\\e matrices cartesian\\e0000_t_0_0.txt"

# Save the matrix to a text file
np.savetxt(file_name, e_0, fmt='%f', delimiter='\t')

# Optional: Display a message
print(f"Matrix saved to {file_name}")

#file_name = "matrix.txt"

# Load the matrix from the text file
r = np.loadtxt(file_name, dtype=float, delimiter='\t')

f=r[:][:]
print(np.max(f-Bx))
