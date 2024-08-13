import __context__

import numpy as np
import matplotlib.pyplot as plt
from matrices.electronic_structure import hamiltonian
from algorithms.chebyshev_nystrom import chebyshev_nystrom
from algorithms.helpers import spectral_transformation, form_spectral_density, gaussian_kernel

np.random.seed(0)

# Load matrix
A = hamiltonian(dim=2)

# Perform spectral transform with A and its eigenvalues
eigvals = np.linalg.eigvalsh(A.toarray())
min_ev, max_ev = eigvals[0], eigvals[-1]
A_st = spectral_transformation(A, min_ev, max_ev)
eigvals_st = spectral_transformation(eigvals, min_ev, max_ev)

# Set parameter
t = np.linspace(-1, 1, 100)
sigma = 0.005
n_Omega = 80
m_list = (np.logspace(1.8, 3.3, 7).astype(int) // 2) * 2

plt.style.use("paper/plots/stylesheet.mplstyle")
colors = ["#648FFF", "#DC267F", "#FFB000"]
markers = ["o", "s", "d"]
labels = ["inconsistent", "consistent", "non-negative"]

# Determine the baseline spectral density
kernel = lambda t, x: gaussian_kernel(t, x, sigma=sigma, n=A.shape[0])
baseline = form_spectral_density(eigvals_st, t, kernel)

error = np.empty((3, len(m_list)))
for j, m in enumerate(m_list):
    estimate = chebyshev_nystrom(A_st, t, m, 0, n_Omega, kernel, consistent=False)
    error[0, j] = 2 * np.mean(np.abs(estimate - baseline))
    estimate = chebyshev_nystrom(A_st, t, m, 0, n_Omega, kernel)
    error[1, j] = 2 * np.mean(np.abs(estimate - baseline))
    estimate = chebyshev_nystrom(A_st, t, m, 0, n_Omega, kernel, nonnegative=True)
    error[2, j] = 2 * np.mean(np.abs(estimate - baseline))

for i in range(3):
    plt.plot(m_list, error[i], color=colors[i], marker=markers[i], label=labels[i])

plt.grid(True, which="both")
plt.ylabel(r"$L^1$-error")
plt.xlabel(r"expansion degree $m$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig("paper/plots/interpolation.pgf", bbox_inches="tight")
