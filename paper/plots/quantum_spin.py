import __context__

import scipy as sp
import numpy as np

from matrices.quantum_spin import hamiltonian, partition_function
from algorithms.chebyshev_nystrom import chebyshev_nystrom
from algorithms.krylov_aware import krylov_aware
from algorithms.helpers import spectral_transformation
import matplotlib.pyplot as plt

N = 20
s = 0.5
h = 0.3
J = 1
A = hamiltonian(N, s, h, J)

# For now, don't execute this
if __name__ == "__main__":
    exit()

E_min, = sp.sparse.linalg.eigsh(A, k=1, which="SA", return_eigenvectors=False, tol=1e-5)
E_max, = sp.sparse.linalg.eigsh(A, k=1, which="LA", return_eigenvectors=False, tol=1e-5)

betas = 1 / np.logspace(-2.5, 3, 50)
Z_true = partition_function(betas, N, h, J, E_min)

A_transformed = spectral_transformation(A, E_min, E_max)
m = 30
n_Omega = 150
n_Psi = 150

function = lambda beta, x: np.exp(-np.multiply.outer(beta, x + 1))
Z_nycheb = chebyshev_nystrom(A_transformed, betas * (E_max - E_min) / 2, m, n_Psi, n_Omega, function, kappa=-1, rcond=1e-10)
Z_nycheb *= np.exp(- betas * (E_min + E_max) / 2)

n_iter = 80
n_reorth = 50
n_Omega = 8
n_Psi = 13

function = lambda beta, x: np.exp(-np.multiply.outer(beta, x - E_min))
Z_krylov = krylov_aware(A, betas, n_iter, n_reorth, n_Omega, n_Psi, function)

plt.style.use("paper/plots/stylesheet.mplstyle")

plt.plot(1 / betas, np.abs(1 - Z_nycheb * A.shape[0] / Z_true), color="#648FFF")  # TODO: CHANGE KERNEL TO NORMALIZED
plt.plot(1 / betas, np.abs(1 - Z_krylov * A.shape[0] / Z_true), color="#FFB000")  # TODO: CHANGE KERNEL TO NORMALIZED

plt.grid(True, which="both")
plt.ylabel(r"relative approximation error")
plt.xlabel(r"inverse temperature $\beta^{-1}$")
plt.xscale("log")
plt.yscale("log")
plt.savefig("paper/plots/quantum_spin.pgf", bbox_inches="tight")
