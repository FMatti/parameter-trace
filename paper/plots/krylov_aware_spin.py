import __context__
from matrices.quantum_spin import hamiltonian, partition_function
from algorithms.chebyshev_nystrom import chebyshev_nystrom
from algorithms.krylov_aware import krylov_aware
from algorithms.helpers import spectral_transformation
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import time

np.random.seed(0)

N = 20
s = 0.5
h = 0.3
J = 1
A = hamiltonian(N, s, h, J)

E_min, = sp.sparse.linalg.eigsh(A, k=1, which="SA", return_eigenvectors=False, tol=1e-5)
E_max, = sp.sparse.linalg.eigsh(A, k=1, which="LA", return_eigenvectors=False, tol=1e-5)

betas = 1 / np.logspace(-2.5, 3, 20)
Z_true = partition_function(betas, N, h, J, E_min)

A_transformed = spectral_transformation(A, E_min, E_max)
m = 100
n_Omega = 40
n_Psi = 40

function = lambda beta, x: np.exp(-np.multiply.outer(beta, x + 1))
t0 = time.time()
Z_nycheb = chebyshev_nystrom(A_transformed, betas * (E_max - E_min) / 2, m, n_Psi, n_Omega, function, kappa=-1, rcond=1e-10)
print(time.time() - t0)
Z_nycheb *= np.exp(- betas * (E_min + E_max) / 2)

n_iter = 30 + 50
n_reorth = 50
n_Omega = 8
n_Psi = 13

function = lambda beta, x: np.exp(-np.multiply.outer(beta, x - E_min))
t0 = time.time()
Z_krylov = krylov_aware(A, betas, n_iter, n_reorth, n_Omega, n_Psi, function)
print(time.time() - t0)

plt.style.use("paper/plots/stylesheet.mplstyle")
colors = ["#FFB000", "#FE6100", "#785EF0", "#648FFF", "#DC267F"]
markers = ["d", "p", "^", "o", "s"]
labels = ["KA I", "KA II", "KA III", "KA IV", "CN++"]

plt.plot(1 / betas, np.abs(1 - Z_krylov / Z_true), label="KA", color="#648FFF", marker="^")
plt.plot(1 / betas, np.abs(1 - Z_nycheb / Z_true), label="CN++", color="#DC267F", marker="s")

plt.grid(True, which="both")
plt.ylabel(r"relative error")
plt.xlabel(r"parameter $\beta^{-1}$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig("paper/plots/krylov_aware_spin.pgf", bbox_inches="tight")
