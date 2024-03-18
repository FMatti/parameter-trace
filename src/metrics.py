"""
Metrics
-------

Error metrics for measuring accuracy of spectral density approximations.
"""

import numpy as np
import scipy as sp


def p_norm(phi, phi_tilde, p=1, relative=True):
    """
    Compute the error in the p-norm of the spectral densities.

    Parameters
    ----------
    phi : np.ndarray of shape (n,)
        The true DOS evaluated at a series of uniformly distributed points.
    phi_tilde : np.ndarray of shape (n,)
        The approximated DOS evaluated at the same points as phi.
    p : int > 0
        The exponent of the p-norm.
    relative : bool
        Whether to compute the relative or absolute error.

    Returns
    -------
    error : float
        The error of the approximated DOS from the actual DOS.
    """
    error = np.sum(np.abs(phi_tilde - phi)**p) ** (1 / p)
    if relative:
        error /= np.sum(np.abs(phi)**p) ** (1 / p)
    return error


# --- Unused implementations ---


def _KL_divergence(phi, phi_tilde, epsilon=0):
    """
    Compute the Kullback-Leibler divergence of the spectral densities.

    Parameters
    ----------
    phi : np.ndarray of shape (n,)
        The true DOS evaluated at a series of uniformly distributed points.
    phi_tilde : np.ndarray of shape (n,)
        The approximated DOS evaluated at the same points as phi.
    epsilon : float > 0
        The threshold for values in phi/phi_tilde to be assumed as zero.

    Returns
    -------
    error : float
        The KL divegence of the approximated DOS from the actual DOS.
    """
    idx = np.logical_and(phi_tilde > epsilon, phi_tilde / phi > epsilon)
    error = np.sum(phi_tilde[idx] * np.log(phi_tilde[idx] / phi[idx]))
    return error


def _KS_distance(phi, phi_tilde):
    """
    Compute the Kolmogorov-Smirnov distance between spectral densities.

    Parameters
    ----------
    phi : np.ndarray of shape (n,)
        The true DOS evaluated at a series of uniformly distributed points.
    phi_tilde : np.ndarray of shape (n,)
        The approximated DOS evaluated at the same points as phi.

    Returns
    -------
    error : float
        The KS distance of the approximated DOS from the actual DOS.
    """
    error = np.max(np.abs(phi - phi_tilde))
    return error


def _JS_distance(phi, phi_tilde, epsilon=0):
    """
    Compute the Jensen-Shannon divergence of the spectral densities.

    Parameters
    ----------
    phi : np.ndarray of shape (n,)
        The true DOS evaluated at a series of uniformly distributed points.
    phi_tilde : np.ndarray of shape (n,)
        The approximated DOS evaluated at the same points as phi.
    epsilon : float > 0
        The threshold for values in phi/phi_tilde to be assumed as zero.

    Returns
    -------
    error : float
        The JS distance between the approximated DOS from the actual DOS.
    """
    phi_mixture = (phi_tilde + phi) / 2
    error = 0.5 * (_KL_divergence(phi, phi_mixture, epsilon) + _KL_divergence(phi_tilde, phi_mixture, epsilon))
    return error


def _Wasserstein_distance(phi, phi_tilde, t=None, normalize=False):
    """
    Compute the Wasserstein distance between the spectral densities.

    Parameters
    ----------
    phi : np.ndarray of shape (n,)
        The true DOS evaluated at a series of uniformly distributed points.
    phi_tilde : np.ndarray of shape (n,)
        The approximated DOS evaluated at the same points as phi.
    t : np.ndarray of shape (n,)
        The points at which the spectral density was evaluated.
    normalize : bool
        Whether to normalize the approximated CDFs before computing the distance.

    Returns
    -------
    error : float
        The Wasserstein distance between the approximated DOS from the actual DOS.
    """
    #phi_CDF = density_to_distribution(phi, t=t, normalize=normalize)
    #phi_tilde_CDF = density_to_distribution(phi_tilde, t=t, normalize=normalize)
    phi_pos = np.clip(phi, 1e-16, None)
    phi_tilde_pos = np.clip(phi_tilde, 1e-16, None)
    error = sp.stats.wasserstein_distance(np.arange(len(phi_pos)), np.arange(len(phi_tilde_pos)), phi_pos, phi_tilde_pos)
    return error
