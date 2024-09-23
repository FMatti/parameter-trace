import __context__

import timeit
import numpy as np

from algorithms.chebyshev_nystrom import chebyshev_expansion
from algorithms.helpers import gaussian_kernel


def time_method(method, parameters, num_times=1000, num_repeats=10):
    """
    Determine the runtime of a method on a set of parameters.

    method : function
        The method which should be timed.
    parameters : dict
        The parameters for the method.
    num_times : int > 0
        The number of runtime computation to consider for averaging.
    num_repeats : int > 0
        The number of times each runtime computation is re-run.

    Returns 
    -------
    mean : float
        The mean runtime in seconds.
    error : float
        The error of the runtime in seconds.
    """
    times = timeit.repeat(lambda: method(**parameters), repeat=num_repeats, number=num_times)
    mean = np.mean(times)
    error = np.std(times)
    return mean, error


def generate_tex_tabular(values, filepath, headline=None, row_labels=None, errors=None, fmt=r"${:.3f}$"):
    """
    Generate a LaTeX tabular from a numpy array.

    Parameters
    ----------
    values : np.ndarray (n, m)
        The values to be displayed in the tabular.
    filepath : str 
        The path to the file in which the table should be stored.
    headline : list of str, int, or float (m,)
        The values in the headline (column labels).
    row_labels : list of str (n,)
        The labels of the rows.
    errors : np.ndarray (n, m)
        The errors associated to the values.
    fmt : str 
        The format in which the values are displayed in the table.
    """
    f = open(filepath, "w")

    num_rows = values.shape[0]
    num_cols = values.shape[1]

    f.write(r"\centering" + "\n")
    f.write(r"\renewcommand{\arraystretch}{1.2}" + "\n")
    f.write(r"\begin{tabular}{@{}" + ("l" if row_labels else "") + num_cols*"c" + r"@{}}" + "\n")
    f.write(r"\toprule" + "\n")
    if headline:
        f.write(r" & ".join(headline) + r"\\" + "\n")
        f.write(r"\midrule" + "\n")

    for i in range(num_rows):
        if row_labels:
            f.write(row_labels[i] + r" & ")
        for j in range(num_cols):
            f.write(fmt.format(values[i, j]))
            if errors is not None:
                f.write(r" $\pm$ " + fmt.format(errors[i, j]))
            if j < num_cols - 1:
                f.write(r" & ")
        f.write(r" \\" + "\n")

    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")

    f.close()


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

    # Can be computed via Fourier transform
    mu = np.real(np.fft.fft(kernel(t, np.cos(theta)), axis=1)[:, :m+1])

    # Rescale the coefficients (as required by the definition)
    mu[:, 0] /= 2 * n_theta
    mu[:, 1:] /= n_theta
    return mu

methods = [chebyshev_coefficients_quadrature, chebyshev_expansion, chebyshev_expansion]
labels = ["quadrature FFT", "DCT", "non-negative DCT"]

n_t = 1000
t = np.arange(-1, 1, n_t)
sigma = 0.005
kernel = lambda t, x: gaussian_kernel(t, x, sigma=sigma)
    
parameters = [{"t": t, "m": 800, "kernel": kernel},
              {"t": t, "m": 1600, "kernel": kernel},
              {"t": t, "m": 2400, "kernel": kernel},
              {"t": t, "m": 3200, "kernel": kernel}]

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
