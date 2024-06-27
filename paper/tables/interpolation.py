import __context__

import numpy as np

from src.interpolation import chebyshev_coefficients, _chebyshev_coefficients_quadrature
from src.kernel import gaussian_kernel
from src.utils import time_method, generate_tex_tabular

methods = [_chebyshev_coefficients_quadrature, chebyshev_coefficients, chebyshev_coefficients]
labels = ["\\cite[algorithm 1]{lin-2017-randomized-estimation}", "\\textbf{DCT}", "\\textbf{DCT (non-negative)}"]

n_t = 1000
t = np.arange(-1, 1, n_t)
sigma = 0.05
n = 1000
g = lambda x: gaussian_kernel(x, n=n, sigma=sigma)
parameters = [{"t": t, "m": 800, "function": g},
              {"t": t, "m": 1600, "function": g},
              {"t": t, "m": 2400, "function": g},
              {"t": t, "m": 3200, "function": g}]

means = np.empty((len(methods), len(parameters)))
errors = np.empty((len(methods), len(parameters)))

for i in range(len(methods)):
    for j in range(len(parameters)):
        if i == 2:
            parameters[j]["nonnegative"] = True
        mean, error = time_method(methods[i], parameters[j], num_times=1000, num_repeats=7)
        means[i, j] = 1e3 * mean
        errors[i, j] = 1e3 * error

headline = ["", r"$m=800$", r"$m=1600$", r"$m=2400$", r"$m=3200$"]

generate_tex_tabular(means, "paper/tables/interpolation.tex", headline, labels, errors, fmt=r"${:.1f}$")