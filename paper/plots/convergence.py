import __context__

import numpy as np
import matplotlib.pyplot as plt
from algorithms.chebyshev_nystrom import chebyshev_nystrom
from algorithms.helpers import spectral_transformation, form_spectral_density, gaussian_kernel
from matrices.electronic_structure import hamiltonian

np.random.seed(0)

# Load matrix
for n in [1, 3]:
    A = hamiltonian(n=n)

    # Perform spectral transform with A and its eigenvalues
    eigvals = np.linalg.eigvalsh(A.toarray())
    min_ev, max_ev = eigvals[0], eigvals[-1]
    A_st = spectral_transformation(A, min_ev, max_ev)
    eigvals_st = spectral_transformation(eigvals, min_ev, max_ev)

    # Set parameter
    t = np.linspace(-1, 1, 100)
    sigma = 0.005
    m = 1000
    n_Vec_list = np.logspace(0.8, 2.8 if n == 1 else 3.4, 7).astype(int) 

    plt.style.use("paper/plots/stylesheet.mplstyle")
    plt.figure(figsize=(3, 3))
    colors = ["#648FFF", "#DC267F", "#FFB000"]
    markers = ["o", "s", "d"]
    labels = [r"$n_{\mathbf{\Omega}} = 0$", r"$n_{\mathbf{\Omega}} = n_{\mathbf{\Psi}}$", r"$n_{\mathbf{\Psi}} = 0$"]

    # Determine the baseline spectral density
    kernel = lambda t, x: gaussian_kernel(t, x, sigma=sigma, n=A.shape[0])
    baseline = form_spectral_density(eigvals_st, t, kernel)

    error = np.empty((3, len(n_Vec_list)))
    for j, n_Vec in enumerate(n_Vec_list):
        estimate = chebyshev_nystrom(A_st, t, m, n_Vec, 0, kernel)
        error[0, j] = 2 * np.mean(np.abs(estimate - baseline))
        estimate = chebyshev_nystrom(A_st, t, m, n_Vec // 2, n_Vec // 2, kernel)
        error[1, j] = 2 * np.mean(np.abs(estimate - baseline))
        estimate = chebyshev_nystrom(A_st, t, m, 0, n_Vec, kernel)
        error[2, j] = 2 * np.mean(np.abs(estimate - baseline))

    for i in range(3):
        plt.plot(n_Vec_list, error[i], color=colors[i], marker=markers[i], label=labels[i])

    if n == 1:
        plt.plot(n_Vec_list, 0.35/n_Vec_list, linestyle="dashed", color="#7a7a7a", alpha=0.5)
        plt.text(8e+1, 5e-3, r"$\mathcal{O}(\varepsilon^{-1})$", color="#7a7a7a")
        plt.plot(n_Vec_list, 0.45/n_Vec_list**(0.5), linestyle="dashed", color="#7a7a7a", alpha=0.5)
        plt.text(6e+1, 9.5e-2, r"$\mathcal{O}(\varepsilon^{-2})$", color="#7a7a7a")

    plt.grid(True)
    plt.ylabel(r"$L^1$-error")
    plt.xlabel(r"estimator size $n_{\mathbf{\Omega}} + n_{\mathbf{\Psi}}$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("paper/plots/convergence-{:d}.pgf".format(n), bbox_inches="tight")
