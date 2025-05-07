import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display


# Define the function f
@np.vectorize
def f(x, y):
    r2 = x**2 + y**2
    val1 = np.exp(-r2) * np.sin(5 * r2)
    val2 = r2
    return val1, val2

# Parallelized function to compute segments
def compute_segment(segment):
    x_segment, y_segment = segment
    z1_segment, z2_segment = f(x_segment, y_segment)
    return z1_segment, z2_segment

def main():
    res = 501  # Define the resolution
    x,y = np.mgrid[-1:1:res*1j, -1:1:res*1j]
    
    num_cores = multiprocessing.cpu_count()  # Get the number of CPU cores
    num_segments = num_cores if res >= num_cores else res  # Determine number of segments
    
    # Divide the grid into segments
    segment_size = res // num_segments
    segments = [(x[:, i:(i+segment_size)], y[:, i:(i+segment_size)]) for i in range(0,x.shape[1],segment_size)]

    # Create a pool of workers
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Compute segments in parallel
        results = pool.map(compute_segment, segments)

    # Combine results
    z1_combined = np.concatenate([result[0] for result in results], axis=1)
    z2_combined = np.concatenate([result[1] for result in results], axis=1)
    
    print(f"z1_shape:  {z1_combined.shape}")
    print(f"z2_shape:  {z2_combined.shape}")
    
    # Plot the results
    plt.imshow(z1_combined, cmap="jet")
    plt.colorbar()
    plt.show()

    plt.imshow(z2_combined, cmap="jet")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
