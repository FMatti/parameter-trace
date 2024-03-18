"""
Utils
-----

Utility functions for the implementations.
"""

import urllib.request
import tarfile
import os
import tempfile
import shutil
import timeit

import numpy as np
import scipy as sp

from src.kernel import gaussian_kernel


def download_matrix(url, save_path="matrices", save_name=None):
    """
    Download a matrix from an online matrix market archive.

    Parameters
    ----------
    url : str
        The URL, e.g. https://www.[...].com/matrix.tar.gz.
    save_path : str
        The path under which the matrix should be saved.
    save_name : str or None
        The filename of the matrix. When None, name is inferred from url.

    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Download the archive containing the matrix
        file_name = os.path.join(temp_dir, "archive.tar.gz")
        archive_file_path, _ = urllib.request.urlretrieve(url, file_name)

        # Open the archive
        with tarfile.open(archive_file_path, "r:gz") as tar:
            # Extract only the ".mtx" files
            mtx_members = [m for m in tar.getmembers() if m.name.endswith(".mtx")]
            tar.extractall(path=temp_dir, members=mtx_members)

            # Convert and save matrices as scipy.sparse.matrix
            for m in mtx_members:
                matrix = sp.io.mmread(os.path.join(temp_dir, m.name))
                if save_name is None:
                    save_name = os.path.splitext(os.path.basename(m.name))[0]
                try:
                    sp.sparse.save_npz(os.path.join(save_path, save_name), matrix)
                except:
                    continue

    finally:
        # Clean up: Delete the temporary directory and its contents
        shutil.rmtree(temp_dir)


def form_spectral_density(eigenvalues, kernel=gaussian_kernel, n=None, a=-1, b=1, n_t=100, grid_points=None, sigma=0.1):
    """
    Compute the (regularized) spectral density of a (small) matrix A at n_t
    evenly spaced grid-points within the interval [a, b].

    Parameters
    ----------
    eigenvalues : np.ndarray of shape (n,)
        The eigenvalues for which the spectral density should be computed.
    n : int > 0
        Size of the matrix A. If None, then the size is assumed to be equal to
        the number of computed eigenvalues.
    a : int or float
        The starting point of the interval within which the density is computed.
    b : int or float > a
        The ending point of the interval within which the density is computed.
    n_t : int > 0
        Number of evenly spaced grid points at which the density is evaluated.
    grid_points : np.ndarray (n_t,)
        (Optional) grid points at which the spectral density should be evaluated.
    sigma : int or float > 0
        Smearing parameter of the spectral density.

    Returns
    -------
    spectral_density : np.ndarray of shape (n_t,)
        The value of the spectral density evaluated at the grid points.
    """
    if grid_points is None:
        grid_points = np.linspace(a, b, n_t)
    spectral_density = np.zeros(len(grid_points))

    for eigenvalue in eigenvalues:
        spectral_density += kernel(
            grid_points - eigenvalue, n=n if n else len(eigenvalues), sigma=sigma
        )

    return spectral_density


def spectral_transformation(A, min_ev=None, max_ev=None, return_ev=False):
    """
    Perform a spectral transformation of a matrix, i.e. transform a spectrum
    contained in (a, b) to (-1, 1).

    Parameters
    ----------
    A : np.ndarray of shape (n, n) or (n,)
        The matrix or vector to be spectrally transformed.
    min_ev : int, float or None
        The starting point of the spectrum. If None is specified, the smallest
        eigenvalue of A is computed and used for min_ev.
    max_ev : int, float or None
        The ending point of the spectrum. If None is specified, the largest
        eigenvalue of A is computed and used for max_ev.
    return_ev : bool
        Whether to return the computed minimum and maximum eigenvalue of A.
    """
    if min_ev is None or max_ev is None:
        eigenvalues = np.linalg.eigvalsh(A if isinstance(A, np.ndarray) else A.toarray())
        if min_ev is None:
            min_ev = np.min(eigenvalues)
        if max_ev is None:
            max_ev = np.max(eigenvalues)
    I = 1
    if isinstance(A, np.ndarray) or isinstance(A, sp.sparse.spmatrix):
        if len(A.shape) == 2:
            I = sp.sparse.eye(*A.shape)
    A_transformed = (2 * A - (min_ev + max_ev) * I) / (max_ev - min_ev)
    
    if return_ev:
        return A_transformed, min_ev, max_ev
    return A_transformed


def continued_fraction(z, a, b):
    """
    Recursively compute the continued fraction of the form

        1 / (z - a[0] - b[0]^2 * 1 / (z - a[1] - b[1] * 1 / (...))).

    Parameters
    ----------
    z : float or complex
        The point at which it is evaluated.
    a : np.ndarray
        Coefficients (diagonal of tridiagonal matrix).
    b : np.ndarray
        Coefficients (off-diagonals of tridiagonal matrix).

    Returns
    -------
    float or complex
        The result of the recursion.
    """
    if len(a) == 1:
        return 1 / (z - a[-1])
    # Here it's - b[0] and not + b[0] like in Lin/Saad/Yang 2016
    return 1 / (z - a[0] - b[0]**2 * continued_fraction(z, a[1:], b[1:]))


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


def theoretical_numerical_rank(n, sigma, epsilon=1e-16):
    """
    Determine the theoretical numerical rank.

    Parameters
    ----------
    n : int > 0
        The size of the matrix, i.e. the number of eigenvalues.
    sigma : int or float > 0
        The smearing-parameters, i.e. the width of the Gaussians.
    epsilon : float > 0
        The value below which singular values are considered equal to zero.
    """
    return n * sigma * np.sqrt(- 2 * np.log(sigma * epsilon * np.sqrt(2 * np.pi)))


def theoretical_chebyshev_degree(sigma, epsilon=1e-16):
    """
    Determine the theoretical degree needed to achieve an error of epsilon.

    Parameters
    ----------
    sigma : int or float > 0
        The smearing-parameters, i.e. the width of the Gaussians.
    epsilon : float > 0
        The tolerated error of the Chebyshev interpolation.
    """
    return - np.log(epsilon * sigma**2) / np.log(1 + sigma)


# --- Unused implementations ---


def _density_to_distribution(phi, t=None, normalize=False):
    if t is None:
        t = np.linspace(-1, 1, len(phi))
    spectral_distribution = np.cumsum(phi) * np.append(0, np.diff(t))

    if normalize:
        spectral_distribution /= spectral_distribution[-1]
    return spectral_distribution


def _form_spectral_distribution(eigenvalues, kernel=gaussian_kernel, n=None, a=-1, b=1, n_t=100, sigma=0.1):
    """
    Compute the (regularized) spectral distribution of a (small) matrix A at n_t
    evenly spaced grid-points within the interval [a, b].

    Parameters
    ----------
    eigenvalues : np.ndarray of shape (n,)
        The eigenvalues for which the spectral distribution should be computed.
    n : int > 0
        Size of the matrix A. If None, then the size is assumed to be equal to
        the number of computed eigenvalues.
    a : int or float
        The starting point of the interval within which the distribution is computed.
    b : int or float > a
        The ending point of the interval within which the distribution is computed.
    n_t : int > 0
        Number of evenly spaced grid points at which the distribution is evaluated.
    sigma : int or float > 0
        Smearing parameter of the spectral distribution.

    Returns
    -------
    spectral_distribution : np.ndarray of shape (n_t,)
        The value of the spectral distribution evaluated at the grid points.
    """
    spectral_density = np.zeros(n_t)
    grid_points = np.linspace(a, b, n_t)

    for eigenvalue in eigenvalues:
        spectral_density += kernel(
            grid_points - eigenvalue, n=n if n else len(eigenvalues), sigma=sigma
        )
    spectral_distribution = np.cumsum(spectral_density)
    return spectral_distribution


def _inverse_spectral_transformation(A, min_ev, max_ev):
    """
    Perform an inverse spectral transformation of a matrix, i.e. transform a
    spectrum contained in (-1, 1) to (a, b).

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        A matrix which was spectrally transformed from (a, b) to (-1, 1).
    min_ev : int, float or None
        The starting point of the spectrum.
    max_ev : int, float or None
        The ending point of the spectrum.
    """
    I = 1
    if isinstance(A, np.ndarray) or isinstance(A, sp.sparse.spmatrix):
        if len(A.shape) == 2:
            I = sp.sparse.eye(*A.shape)
    return (max_ev - min_ev) / 2 * A + (min_ev + max_ev) / 2 * I


def _verify_parameters(n, sigma, m, n_v, n_v_tilde=None, epsilon=1e-16):
    M_min = theoretical_chebyshev_degree(sigma, epsilon)
    N_v_min = theoretical_numerical_rank(n, sigma, epsilon)

    if m < M_min:
        print("Degree of Chebyshev polynomial too low.")
    if n_v < N_v_min:
        print("Number of random vectors for low-rank approximation too low.")
