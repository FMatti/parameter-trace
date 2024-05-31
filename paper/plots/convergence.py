import __context__

import numpy as np
import scipy as sp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from src.simple import spectral_density, spectral_transformation, form_spectral_density, gaussian_kernel

np.random.seed(0)

# Load matrix
A = sp.sparse.load_npz("matrices/ModES3D_1.npz")

# Perform spectral transform with A and its eigenvalues
eigvals = np.linalg.eigvalsh(A)#.toarray())
min_ev, max_ev = eigvals[0], eigvals[-1]
A_st = spectral_transformation(A, min_ev, max_ev)
eigvals_st = spectral_transformation(eigvals, min_ev, max_ev)

# Set parameter
t = np.linspace(-1, 1, 100)
sigma = 0.2
m = 500
n_Vec_list = np.logspace(0.95, 2.6, 2).astype(int) 

plt.style.use("paper/plots/stylesheet.mplstyle")
colors = cm.magma(np.arange(5) / 4.5)
markers = ["o", "^", "s"]
labels = [r"$n_{\mathbf{\Omega}} = 0$",
          r"$n_{\mathbf{\Psi}} = n_{\mathbf{\Omega}}$",
          r"$n_{\mathbf{\Psi}} = 0$"]

# Determine the baseline spectral density
kernel = lambda x: gaussian_kernel(x, sigma=sigma)
baseline = form_spectral_density(eigvals_st, t, kernel)

error = np.empty((3, len(n_Vec_list)))
for j, n_Vec in enumerate(n_Vec_list):
    estimate = spectral_density(A_st, t, m, n_Vec, 0, kernel)
    error[0, j] = 2 * np.mean(np.abs(estimate - baseline))
    estimate = spectral_density(A_st, t, m, n_Vec // 2, n_Vec // 2, kernel)
    error[1, j] = 2 * np.mean(np.abs(estimate - baseline))
    estimate = spectral_density(A_st, t, m, 0, n_Vec, kernel)
    error[2, j] = 2 * np.mean(np.abs(estimate - baseline))

for i in range(3):
    plt.plot(n_Vec_list, error[i], color=colors[i], marker=markers[i], label=labels[i])

plt.plot(n_Vec_list, 0.35/n_Vec_list, linestyle="dashed", color="#7a7a7a", alpha=0.5)
plt.text(8e+1, 5e-3, r"$\mathcal{O}(\varepsilon^{-1})$", color="#7a7a7a")
plt.plot(n_Vec_list, 0.45/n_Vec_list**(0.5), linestyle="dashed", color="#7a7a7a", alpha=0.5)
plt.text(6e+1, 9.5e-2, r"$\mathcal{O}(\varepsilon^{-2})$", color="#7a7a7a")

plt.grid()
plt.ylabel(r"$L^1$ error")
plt.xlabel(r"estimator size $n_{\mathbf{\Psi}} + n_{\mathbf{\Omega}}$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
