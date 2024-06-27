import __context__

import numpy as np

from src.simple import gaussian_kernel, chebyshev_expansion
from src.utils import time_method, generate_tex_tabular

def chebyshev_coefficients_quadrature(t, m, kernel, n_theta=None):
    """
    Delta-Gauss-Chebyshev polynomial expansion.

    Parameters
    ----------
    t : int, float, list, or np.ndarray of shape (n,)
        Point(s) where the expansion should be evaluated.
    m : int > 0
        Degree of the Chebyshev polynomial.
    kernel : callable
        The kernel used to regularize the spectral density.
    n_theta : int > M
        The (half) number of integration points.

    Returns
    -------
    mu : np.ndarray of shape (N_t, M + 1)
        The coefficients of the Chebyshev polynomials. Format: mu[t, l].

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 1.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # If t is a scalar, we convert it to a 1d array to make computation work
    if not isinstance(t, np.ndarray):
        t = np.array(t).reshape(-1)

    # If not specified, take minimum number of quadrature nodes
    if n_theta is None:
        n_theta = 2*(m + 1)

    theta = np.arange(2 * n_theta) * np.pi / n_theta

    # Can be computed via Fourier transform:
    t_minus_theta = np.subtract.outer(t, np.cos(theta))
    mu = np.real(np.fft.fft(kernel(t_minus_theta), axis=1)[:, :m+1])

    # Rescale the coefficients (as required by the definition)
    mu[:, 0] /= 2 * n_theta
    mu[:, 1:] /= n_theta
    return mu

methods = [chebyshev_coefficients_quadrature, chebyshev_expansion, chebyshev_expansion]
labels = ["oversampled FFT", "DCT", "non-negative DCT"]

n_t = 1000
t = np.arange(-1, 1, n_t)
sigma = 0.005
g = lambda x: gaussian_kernel(x, sigma=sigma)
parameters = [{"t": t, "m": 800, "kernel": g},
              {"t": t, "m": 1600, "kernel": g},
              {"t": t, "m": 2400, "kernel": g},
              {"t": t, "m": 3200, "kernel": g}]

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
