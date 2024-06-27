import __context__

import numpy as np
import matplotlib.pyplot as plt

from src.algorithms import NC
from src.matrices import ModES3D
from src.plots import compute_spectral_density_errors, plot_spectral_density_errors

plt.style.use("paper/plots/stylesheet.mplstyle")

np.random.seed(0)

methods = NC
labels = ["interpolation", "squaring", "non-negative"]
fixed_parameters = [{"n_v": 80, "sigma": 0.05, "square_coefficients": "interpolation"},
                    {"n_v": 80, "sigma": 0.05, "square_coefficients": "transformation"},
                    {"n_v": 80, "sigma": 0.05, "square_coefficients": "transformation", "nonnegative": True}]
variable_parameters = "m"
variable_parameters_values = (np.logspace(2.3, 3.5, 7).astype(int) // 2) * 2

A = ModES3D(dim=2)
colors = ["#FFB000", "#648FFF", "#DC267F"]
markers = ["o", "^", "s"]
spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100)

fig, ax = plt.subplots(figsize=(4, 3))
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["square_coefficients", "n_v", "sigma", "nonnegative"], colors=colors, error_metric_name="$L^1$ error", x_label="expansion degree $m$", markers=markers, ax=ax)

plt.savefig("paper/plots/interpolation.pgf", bbox_inches="tight")