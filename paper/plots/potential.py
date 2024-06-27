import __context__

import matplotlib.pyplot as plt

from src.matrices import regular_grid, periodic_gaussian_well

plt.style.use("paper/plots/stylesheet.mplstyle")

# Define grid parameters
N = 100
L = 6
alpha = -4.0
beta = 2.0

# Plot periodic Gaussian wells for different number of repetitions
ns = [1, 2, 5]
for n in ns:
    fig, ax = plt.subplots(figsize=(2, 1.75))
    #ax.set_xlim([0, 1])
    #ax.set_ylim(-11, 80)
    grid_points = regular_grid(a=0, b=n*L, N=N, dim=2)
    grid_values = periodic_gaussian_well(grid_points, L=L, n=n, var=beta**2, scaling_factor=alpha)
    plt.contourf(grid_points[:, 0].reshape(N, N), grid_points[:, 1].reshape(N, N), grid_values.reshape(N, N), levels=8, cmap="plasma_r")
    plt.savefig("paper/plots/gaussian-well-{:d}.pgf".format(n), bbox_inches="tight")
