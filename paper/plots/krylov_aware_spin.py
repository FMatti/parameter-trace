import __context__

import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from algorithms.chebyshev_nystrom import chebyshev_nystrom
from algorithms.krylov_aware import krylov_aware
from algorithms.helpers import spectral_transformation, generate_tex_tabular
from matrices.quantum_spin import hamiltonian, partition_function

np.random.seed(0)

# Don't execute this because memory overflow on GitHub due to private repository
#if __name__ == "__main__":
#    quit()

N = 20
s = 0.5
h = 0.3
J = 1
A = hamiltonian(N, s, h, J)

E_min, = sp.sparse.linalg.eigsh(A, k=1, which="SA", return_eigenvectors=False, tol=1e-5)
E_max, = sp.sparse.linalg.eigsh(A, k=1, which="LA", return_eigenvectors=False, tol=1e-5)

A_st = spectral_transformation(A, E_min, E_max)

# Set parameter
n_Omega_list = np.array([8, 0, 8, 4, 40])  # b
n_Psi_list = np.array([0, 13, 13, 6, 40])  # m
q_list = np.array([30, 0, 30, 30, None])  # q
r_list = np.array([50, 50, 50, 50, None])  # n
m = 50

plt.style.use("paper/plots/stylesheet.mplstyle")
plt.figure(figsize=(3, 3))
colors = ["#FFB000", "#FE6100", "#785EF0", "#648FFF", "#DC267F"]
markers = ["d", "p", "^", "o", "s"]
labels = ["KA (i)", "KA (ii)", "KA (iii)", "KA (iv)", "CN++"]

# Determine the baseline spectral density
betas = 1 / np.logspace(-2.5, 3, 16)
baseline = partition_function(betas, N, h, J, E_min)

error = np.empty((len(n_Psi_list), len(betas)))
times = np.zeros(len(n_Psi_list))

for i, (n_Psi, n_Omega, q, r) in enumerate(zip(n_Psi_list, n_Omega_list, q_list, r_list)):
    t0 = time.time()
    if i < len(n_Psi_list) - 1:
        function = lambda beta, x: np.exp(-np.multiply.outer(beta, x - E_min))
        estimate = krylov_aware(A, betas, r + q, q, n_Omega, n_Psi, function)
    else:
        function = lambda beta, x: np.exp(-np.multiply.outer(beta, x + 1))
        estimate = chebyshev_nystrom(A_st, betas * (E_max - E_min) / 2, m, n_Psi, n_Omega, function, kappa=-1, rcond=1e-10)
        estimate *= np.exp(- betas * (E_min + E_max) / 2)
    error[i] = np.abs(1 - estimate / baseline)
    times[i] = (time.time() - t0) / len(n_Psi_list)

for i in range(len(n_Psi_list)):
    plt.plot(1 / betas, error[i], color=colors[i], marker=markers[i], label=labels[i])

plt.grid(True, which="both")
plt.ylabel(r"relative error")
plt.xlabel(r"temperature parameter $\beta^{-1}$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig("paper/plots/krylov_aware_spin.pgf", bbox_inches="tight")

headline = ["", r"$n_{\mtx{\Omega}}$", r"$n_{\mtx{\Psi}}$", r"$q$", r"$r$", r"time (s)"]
fmt = [r"${:0.0f}$", r"${:0.0f}$", r"${:0.0f}$", r"${:0.0f}$", r"${:.2f}$"]
values = np.vstack((n_Omega_list[:-1], n_Psi_list[:-1], q_list[:-1], r_list[:-1], times[:-1])).T

generate_tex_tabular(values, "paper/tables/krylov_aware_spin_KA.tex", headline, labels[:-1], fmt=fmt)

headline = ["", r"$n_{\mtx{\Omega}}$", r"$n_{\mtx{\Psi}}$", r"$m$", r"time (s)"]
fmt = [r"${:0.0f}$", r"${:0.0f}$", r"${:0.0f}$", r"${:.2f}$"]
values = np.vstack((n_Omega_list[-1], n_Psi_list[-1], m, times[-1])).T

generate_tex_tabular(values, "paper/tables/krylov_aware_spin_CN.tex", headline, labels[-1:], fmt=fmt)
