import __context__

import numpy as np
import matplotlib.pyplot as plt
from algorithms.chebyshev_nystrom import chebyshev_nystrom
from algorithms.helpers import spectral_transformation, form_spectral_density, gaussian_kernel
from matrices.electronic_structure import hamiltonian

np.random.seed(0)

# Load matrix
A = hamiltonian(dim=3)

# Perform spectral transform with A and its eigenvalues
eigvals = np.linalg.eigvalsh(A.toarray())
min_ev, max_ev = eigvals[0], eigvals[-1]
A_st = spectral_transformation(A, min_ev, max_ev)
eigvals_st = spectral_transformation(eigvals, min_ev, max_ev)

# Set parameter
t = np.linspace(-1, 1, 500)
sigma = 0.004
m = 2000
n_Omega = 80

plt.style.use("paper/plots/stylesheet.mplstyle")
colors = ["#648FFF", "#DC267F", "#FFB000"]
labels = ["baseline", "without zero-check", "with zero-check"]
kappa_list = [-1, 1e-5]

# Determine the baseline spectral density
kernel = lambda t, x: gaussian_kernel(t, x, sigma=sigma, n=A.shape[0])
baseline = form_spectral_density(eigvals_st, t, kernel)

estimate = [baseline]
for i, kappa in enumerate(kappa_list):
    estimate.append(chebyshev_nystrom(A_st, t, m, 0, n_Omega, kernel, kappa=kappa))

for i in range(3):
    plt.plot(t, estimate[i], color=colors[i], label=labels[i])

plt.grid(True, which="both")
plt.ylabel(r"smoothed spectral density $\phi_{\sigma}(t)$")
plt.xlabel(r"spectral parameter $t$")
plt.legend()
plt.savefig("paper/plots/zerocheck.pgf", bbox_inches="tight")
