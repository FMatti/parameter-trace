import __context__

import time
import numpy as np
import matplotlib.pyplot as plt
from algorithms.chebyshev_nystrom import chebyshev_nystrom
from algorithms.krylov_aware import krylov_aware
from algorithms.helpers import spectral_transformation, form_spectral_density, gaussian_kernel, generate_tex_tabular
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
n_Omega_list = np.array([ 10, 20, 10, 10, 20])    # b
n_Psi_list = np.array([ 60, 30, 30, 30, 20])    # m
q_list = np.array([20, 20, 40, 20, None])  # q
n_list = np.array([10, 10, 10, 20, None])  # n
m = 2000

plt.style.use("paper/plots/stylesheet.mplstyle")
plt.figure(figsize=(3, 3))
colors = ["#FFB000", "#FE6100", "#785EF0", "#648FFF", "#DC267F"]
markers = ["d", "p", "^", "o", "s"]
labels = ["KA (I)", "KA (II)", "KA (III)", "KA (IV)", "CN++"]

error = np.empty((len(n_Psi_list), len(sigma_list)))
times = np.zeros(len(n_Psi_list))
for j, sigma in enumerate(sigma_list):

    # Determine the baseline spectral density
    kernel = lambda t, x: gaussian_kernel(t, x, sigma=sigma, n=A.shape[0])
    baseline = form_spectral_density(eigvals_st, t, kernel)

    for i, (n_Psi, n_Omega, q, n) in enumerate(zip(n_Psi_list, n_Omega_list, q_list, n_list)):
        t0 = time.time()
        if i < len(n_Psi_list) - 1:
            estimate = krylov_aware(A_st, t, n + q, q, n_Psi, n_Omega, kernel)
        else:
            estimate = chebyshev_nystrom(A_st, t, m, n_Psi, n_Omega, kernel)
        error[i, j] = 2 * np.mean(np.abs(estimate - baseline))
        times[i] += (time.time() - t0) / len(n_Psi_list)

for i in range(len(n_Psi_list)):
    plt.plot(sigma_list, error[i], color=colors[i], marker=markers[i], label=labels[i])

plt.grid(True, which="both")
plt.ylabel(r"$L^1$-error")
plt.xlabel(r"smoothing parameter $\sigma$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig("paper/plots/krylov_aware_density.pgf", bbox_inches="tight")

headline = ["", r"$n_{\mtx{\Omega}}$", r"$n_{\mtx{\Psi}}$", r"$q$", r"$r$", r"time (s)"]
fmt = [r"${:0.0f}$", r"${:0.0f}$", r"${:0.0f}$", r"${:0.0f}$", r"${:.2f}$"]
values = np.vstack((n_Omega_list[:-1], n_Psi_list[:-1], q_list[:-1], n_list[:-1], times[:-1])).T

generate_tex_tabular(values, "paper/tables/krylov_aware_density_KA.tex", headline, labels[:-1], fmt=fmt)

headline = ["", r"$n_{\mtx{\Omega}}$", r"$n_{\mtx{\Psi}}$", r"$m$", r"time (s)"]
fmt = [r"${:0.0f}$", r"${:0.0f}$", r"${:0.0f}$", r"${:.2f}$"]
values = np.vstack((n_Omega_list[-1], n_Psi_list[-1], m, times[-1])).T

generate_tex_tabular(values, "paper/tables/krylov_aware_density_CN.tex", headline, labels[-1:], fmt=fmt)
