"""
chebyshev_nystrom.py
--------------------

Implementation of the Chebyshev-Nyström++ method for spectral density estimation.
"""

import numpy as np
import scipy as sp

from .helpers import gaussian_kernel

def chebyshev_expansion(kernel, t, m, nonnegative=False):
    """
    Chebyshev expansion of a kernel.

    Parameters
    ----------
    kernel : kernel
        The kernel used to regularize the spectral density.
    t : int, float, list, or np.ndarray of shape (n_t,)
        Point(s) where the expansion should be evaluated.
    m : int > 0
        Degree of the Chebyshev polynomial.
    nonnegative : bool
        Force interpolant to be non-negative (for non-negative functions).

    Returns
    -------
    mu : np.ndarray of shape (n_t, m + 1)
        The coefficients of the Chebyshev polynomials. Format: mu[t, l].
    """
    t = np.asarray(t, dtype=np.float64)

    if nonnegative:
        m = m // 2
        kernel_ = kernel # Avoid recursion
        kernel = lambda t, x: np.sqrt(kernel_(t, x))

    # Compute the coefficients mu for all t and l simultaneously with DCT
    chebyshev_nodes = np.cos(np.arange(m + 1) * np.pi / m)
    mu = sp.fft.idct(kernel(t, chebyshev_nodes), type=1)

    # Rescale coefficients due to type-1 DCT convention
    mu[..., 1:-1] *= 2

    # Square Chebyshev expansion
    if nonnegative:
        mu = square_chebyshev_expansion(mu)

    return mu

def square_chebyshev_expansion(mu):
    """
    Square a Chebyshev expansion with coefficients mu.

    Parameters
    ----------
    mu : list or np.ndarray of shape (n_t, m + 1)
        The coefficients of the Chebyshev polynomials. Format: mu[t, l].

    Returns
    -------
    nu : np.ndarray of shape (n_t, 2 * m + 1)
        The coefficients of the squared Chebyshev polynomials. Format: nu[t, l].
    """
    mu = np.asarray(mu, dtype=np.float64)

    # Zero-pad last axis of the coefficients with m zeros
    pad_config = [*([(0, 0)] * (mu.ndim - 1)), (0, mu.shape[-1] - 1)]
    mu = np.pad(mu, pad_config, mode="constant")

    # Rescale coefficients due to type-2 DCT convention
    mu[..., 1:-1] /= 2

    # Chain DCT with an inverse DCT to compute exponentiated coefficients
    nu = sp.fft.idct(sp.fft.dct(mu, type=1) ** 2, type=1)

    # Rescale coefficients due to type-2 DCT convention
    nu[..., 1:-1] *= 2

    return nu

def chebyshev_nystrom(A, t=0, m=100, n_Psi=10, n_Omega=10, kernel=gaussian_kernel, nonnegative=False, consistent=True, kappa=1e-5, rcond=1e-5, seed=0):
    """
    Chebyshev-Nyström++ method for estimating the spectral density.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Symmetric matrix with eigenvalues between (-1, 1).
    t : np.ndarray (n_t,)
        A set of points at which the DOS is to be evaluated.
    m : int > 0
        Degree of Chebyshev the polynomial.
    n_Psi : int > 0
        Number of queries with Girard-Hutchinson estimator.
    n_Omega : int > 0
        Size of sketching matrix in Nyström approximation.
    kernel : callable
        Smoothing kernel.
    nonnegative : bool
        Use non-negative Chebyshev expansion.
    kappa : float > 0
        The threshold on the Hutchinson estimate of g_sigma. If it is below this
        value, instead of solving the possibly ill-conditioned generalized
        eigenvalue problem, we set the spectral density at that point to zero.
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    phi : np.ndarray
        Approximations of the spectral density at the points t.

    """
    # Seed the random number generator
    np.random.seed(seed)

    # Convert evaluation point(s) to numpy array
    if not isinstance(t, np.ndarray):
        t = np.array(t).reshape(-1)

    # Determine size of matrix i.e. number of eigenvalues 
    n = A.shape[0]
    n_t = t.shape[0]

    # Compute coefficients of Chebyshev expansion of the smoothing kernel
    mu = chebyshev_expansion(kernel, t, m, nonnegative=nonnegative)
    if consistent:
        nu = square_chebyshev_expansion(mu)
    else:
        nu = chebyshev_expansion(lambda t, x: kernel(t, x) ** 2, t, 2 * m, nonnegative=nonnegative)

    # Generate Gaussian random matrices
    Omega = np.random.randn(n, n_Omega)
    Psi = np.random.randn(n, n_Psi)

    # Initialize Chebyshev recurrence and helper matrices
    V_1, V_2, V_3 = np.zeros((n, n_Omega)), Omega, np.zeros((n, n_Omega))
    W_1, W_2, W_3 = np.zeros((n, n_Psi)), Psi, np.zeros((n, n_Psi))
    K_1, K_2 = np.zeros((n_t, n_Omega, n_Omega)), np.zeros((n_t, n_Omega, n_Omega))
    L_1, l_2 = np.zeros((n_t, n_Omega, n_Psi)), np.zeros(n_t)

    for l in range(2 * m + 1):

        # Helper quantities
        X, Y = Omega.T @ V_2, Omega.T @ W_2
        z = np.sum(np.multiply(Psi,  W_2))

        for i in range(n_t):
            # Accumulation
            if l <= m:
                K_1[i] += mu[i, l] * X
                l_2[i] += mu[i, l] * z
                L_1[i] += mu[i, l] * Y
            K_2[i] += nu[i, l] * X

        # Chebyshev recurrence
        V_3, W_3 = (2 - (l == 0)) * (A @ V_2) - V_1, (2 - (l == 0)) * (A @ W_2) - W_1
        V_1, W_1 = V_2, W_2
        V_2, W_2 = V_3, W_3

    # Evaluate spectral density
    phi = np.zeros(n_t)
    for i in range(n_t):

        # Early-stopping
        if n_Omega and np.trace(K_1[i]) <= kappa * n_Omega:
            continue

        phi[i] = np.trace(np.linalg.lstsq(K_1[i], K_2[i], rcond=rcond)[0])
        if n_Psi > 0:
            phi[i] += (l_2[i] - np.trace(L_1[i].T @ np.linalg.lstsq(K_1[i], L_1[i], rcond=rcond)[0])) / n_Psi

    return phi
