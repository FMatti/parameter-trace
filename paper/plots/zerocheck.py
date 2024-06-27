#import __context__
#
#import numpy as np
#import scipy as sp
#import matplotlib.pyplot as plt
#
#from src.simple import spectral_density
#from src.plots import compute_spectral_densities, plot_spectral_densities
#
#import matplotlib
#matplotlib.rc("text", usetex=True)
#matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
#matplotlib.rcParams["font.family"] = r"serif"
#matplotlib.rcParams["font.serif"] = r"Palatino"
#matplotlib.rcParams["font.size"] = 12
#
#np.random.seed(0)
#
#methods = spectral_density
#labels = ["no non-zero check", "non-zero check"]
#parameters = [{"m": 2000, "sigma": 0.05, "n_v": 80, "kappa": -1},
#              {"m": 2000, "sigma": 0.05, "n_v": 80, "kappa": 1e-5}]
#
#A = sp.sparse.load_npz("matrices/ModES3D_1.npz")
#colors = ["#648FFF", "#DC267F", "#FFB000"]
#spectral_densities = compute_spectral_densities(A, methods, labels, parameters, add_baseline=False, n_t=500)
#
#fig, ax = plt.subplots(figsize=(6, 3))
#plot_spectral_densities(spectral_densities, parameters, variable_parameter="kappa", ignored_parameters=["m", "sigma", "n_v", "kappa"], colors=colors, ax=ax)
#
#plt.savefig("thesis/plots/short_circuit_mechanism.pgf", bbox_inches="tight")
#
#
import __context__

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from src.simple import spectral_density, spectral_transformation, form_spectral_density, gaussian_kernel
from src.matrices import hamiltonian

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
kernel = lambda x: gaussian_kernel(x, sigma=sigma)
baseline = form_spectral_density(eigvals_st, t, kernel)

estimate = [baseline]
for i, kappa in enumerate(kappa_list):
    estimate.append(spectral_density(A_st, t, m, 0, n_Omega, kernel, kappa=kappa))

for i in range(3):
    plt.plot(t, estimate[i], color=colors[i], label=labels[i])

plt.grid(True, which="both")
plt.ylabel(r"smoothed spectral density $\phi_{\sigma}(t)$")
plt.xlabel(r"spectral parameter $t$")
plt.legend()
plt.savefig("paper/plots/zerocheck.pgf", bbox_inches="tight")
