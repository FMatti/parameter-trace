"""
helpers.py
----------

Collection of helper functions for the numerical experiments.
"""

import numpy as np
import scipy as sp

def spectral_transformation(A, min_ev=None, max_ev=None):
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
    """
    I = 1
    if len(A.shape) == 2:
        if isinstance(A, sp.sparse.spmatrix):
            I = sp.sparse.eye(*A.shape)
        if isinstance(A, np.ndarray):
            I = np.eye(*A.shape)

    A_st = (2 * A - (min_ev + max_ev) * I) / (max_ev - min_ev)

    return A_st

def gaussian_kernel(t, x=None, sigma=0.1, n=1):
    """
    Gaussian kernel.

    Parameters
    ----------
    t : int, float, or np.ndarray of shape (n, m)
        Parameter value(s) where the Gaussian should be evaluated.
    x : int, float, or np.ndarray of shape (n, m)
        Value(s) outer-subtracted from paramter value(s) 
    sigma : int or float
        Standard deviation of the Gaussian.
    n : int
        Normalization "g(t, x) / n" for simpler spectral density computations.
        Here, n is the size of the matrix.

    Returns
    -------
    g(t, x) : np.ndarray of shape (n, dim)
        The Gaussian kernel evaulated at all combinations of points t - x
    """
    if x is not None:
        t = np.subtract.outer(t, x)
    return np.exp(- t**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma * n)

def form_spectral_density(eigvals, t, kernel=gaussian_kernel):
    """
    Compute the (regularized) spectral density of a (small) matrix A at n_t
    evenly spaced grid-points within the interval [a, b].

    Parameters
    ----------
    eigvals : np.ndarray of shape (n,)
        The eigenvalues for which the spectral density should be formed.
    t: np.ndarray (n_t,)
        Parameter values at which the spectral density should be evaluated.
    kernel : function
        Smoothing kernel.

    Returns
    -------
    spectral_density : np.ndarray of shape (n_t,)
        The value of the spectral density evaluated at the grid points.
    """
    return kernel(t, eigvals).sum(axis=1)
