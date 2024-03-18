"""
Kernels
-------

Collection of smoothing kernels
"""

import numpy as np


def gaussian_kernel(x, n=1, sigma=1.0):
    """
    Gaussian kernel.

    Parameters
    ----------
    x : int, float, or np.ndarray of shape (n, m)
        Point(s) where the Gaussian should be evaluated.
    n : int
        Number of eigenvalues (i.e. size of matrix).
    sigma : int or float
        Standard deviation of the Gaussian.

    Returns
    -------
    g(x) : np.ndarray of shape (n, dim)
        The Gaussian kernel evaulated at all points x.
    """
    normalization = n * np.sqrt(2 * np.pi) * sigma
    exponent = - x**2 / (2 * sigma**2)
    return np.exp(exponent) / normalization


def lorentzian_kernel(x, n=1, sigma=1.0):
    """
    Kernel modeled after the Cauchy distribution.

    Parameters
    ----------
    x : int, float, or np.ndarray of shape (n, m)
        Point(s) where the Gaussian should be evaluated.
    n : int
        Number of eigenvalues (i.e. size of matrix).
    sigma : int or float
        Standard deviation of the Gaussian.

    Returns
    -------
    g(x) : np.ndarray of shape (n, dim)
        The Cauchy kernel evaulated at all points x.

    Reference
    ---------
    [1] https://en.wikipedia.org/wiki/Cauchy_distribution#Probability_density_function_(PDF)
    """
    return sigma / (n * np.pi * (x**2 + sigma**2))


# --- Unused implementations ---


def _bump_kernel(x, n=1, sigma=1.0):
    """
    Kernel modeled after the Cauchy distribution.

    Parameters
    ----------
    x : int, float, or np.ndarray of shape (n, m)
        Point(s) where the Gaussian should be evaluated.
    n : int
        Number of eigenvalues (i.e. size of matrix).
    sigma : int or float
        Standard deviation of the Gaussian.

    Returns
    -------
    g(x) : np.ndarray of shape (n, dim)
        The Cauchy kernel evaulated at all points x.

    Reference
    ---------
    [1] https://en.wikipedia.org/wiki/Cauchy_distribution#Probability_density_function_(PDF)
    """
    normalization = sigma * n * 0.443994
    exponent = - 1 / (1 - np.abs(x / sigma)**2)
    result = np.where(np.abs(x) < sigma, np.exp(exponent) / normalization, 0)
    return result


def _epanechnikov_kernel(x, n=1, sigma=1.0):
    normalization = 3 / (4 * sigma * n)
    function = 1 - (x / sigma)**2
    result = np.where(np.abs(x) < sigma, function * normalization, 0)
    return result


def _quartic_kernel(x, n=1, sigma=1.0):
    normalization = 15 / (16 * sigma * n)
    function = (1 - (x / sigma)**2) ** 2
    result = np.where(np.abs(x) < sigma, function * normalization, 0)
    return result


def _tricube_kernel(x, n=1, sigma=1.0):
    normalization = 70 / (81 * sigma * n)
    function = (1 - np.abs(x / sigma)**3) ** 3
    result = np.where(np.abs(x) < sigma, function * normalization, 0)
    return result


def _cosine_kernel(x, n=1, sigma=1.0):
    normalization = np.pi / (4 * sigma * n)
    function = np.cos(np.pi * x / (2 * sigma))
    result = np.where(np.abs(x) < sigma, function * normalization, 0)
    return result


def _logistic_kernel(x, n=1, sigma=1.0):
    normalization = 1 / (sigma * n)
    function = 1 / (np.exp(x / sigma) + 2 + np.exp(-x / sigma))
    result = function * normalization
    return result


def _sigmoid_kernel(x, n=1, sigma=1.0):
    normalization = 2 / (np.pi * sigma * n)
    function = 1 / (np.exp(x / sigma) + np.exp(-x / sigma))
    result = function * normalization
    return result

