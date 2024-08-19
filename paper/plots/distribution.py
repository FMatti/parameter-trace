import __context__

import numpy as np
import matplotlib.pyplot as plt
from algorithms.chebyshev_nystrom import chebyshev_nystrom
from algorithms.helpers import spectral_transformation, form_spectral_density, gaussian_kernel
from matrices.electronic_structure import hamiltonian

np.random.seed(0)

# Load matrix
A = hamiltonian()

# Perform spectral transform with A and its eigenvalues
eigvals = np.linalg.eigvalsh(A.toarray())
min_ev, max_ev = eigvals[0], eigvals[-1]
A_st = spectral_transformation(A, min_ev, max_ev)
eigvals_st = spectral_transformation(eigvals, min_ev, max_ev)

# Set parameter
t = np.linspace(-1, 1, 100)
sigma_list = np.logspace(-3.5, -1.0, 7)
n_Vec = 80
n_Psi_list = np.arange(n_Vec + 1, step=20).astype(np.int64)
n_Omega_list = n_Vec - n_Psi_list

plt.style.use("paper/plots/stylesheet.mplstyle")
colors = ["#FFB000", "#FE6100", "#DC267F", "#785EF0", "#648FFF"]
markers = ["d", "p", "s", "^", "o"]
labels = [r"$n_{\mathbf{\Psi}} = " + "{}$, ".format(n_Psi) + r"$n_{\mathbf{\Omega}} = " + "{}$".format(n_Omega) for n_Psi, n_Omega in zip(n_Psi_list, n_Omega_list)]

error = np.empty((len(n_Psi_list), len(sigma_list)))
for j, sigma in enumerate(sigma_list):

    # Determine the baseline spectral density
    kernel = lambda t, x: gaussian_kernel(t, x, sigma=sigma, n=A.shape[0])
    baseline = form_spectral_density(eigvals_st, t, kernel)
    m = int(16 / sigma)  # Double-check this, i.e. use error estimate for this

    for i, (n_Psi, n_Omega) in enumerate(zip(n_Psi_list, n_Omega_list)):
        estimate = chebyshev_nystrom(A_st, t, m, n_Psi, n_Omega, kernel)
        error[i, j] = 2 * np.mean(np.abs(estimate - baseline))

for i in reversed(range(len(n_Psi_list))):
    plt.plot(sigma_list, error[i], color=colors[i], marker=markers[i], label=labels[i])

plt.grid(True, which="both")
plt.ylabel(r"$L^1$-error")
plt.xlabel(r"smoothing parameter $\sigma$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig("paper/plots/distribution.pgf", bbox_inches="tight")
