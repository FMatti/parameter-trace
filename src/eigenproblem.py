"""
Eigenproblem
------------

Different solution strategies for solving the generalized eigenvalue problem

    K_2 * C = K_1 * C * Xi
"""

import numpy as np
import scipy as sp


def generalized_eigenproblem_standard(K_2, K_1, n, sigma=1.0, zeta=1e-7, eta=1e-3):
    """
    Solve the generalized eigenvalue problem as in spectrum sweeping method [1].

    Parameters
    ----------
    K_2 : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_2 = Z* Z = W* P^2 W.
    K_1 : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_1 = W* Z = W* P W.
    n : int > 0
        Size of original problem.
    sigma : int or float > 0
        Smearing parameter.
    zeta : int or float in (0, 1]
        Truncation parameter.
    eta : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma. 

    Returns
    -------
    xi : np.ndarray
        The generalized eigenvalues.
    D : np.ndarray
        Matrix which can be used to reconstruct xi.

    References
    ----------
    [1] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 4.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Compute eigenvalue decomposition
    gamma, W = np.linalg.eigh(K_1)

    # Only keep large enough eigenvalues
    idx = np.where(gamma >= zeta * np.max(gamma))[0].flatten()
    gamma_1_invsqrt = gamma[idx] ** (-0.5)
    W_1 = W[:, idx]

    # Form eigenvalue problem
    LHS = np.outer(gamma_1_invsqrt, gamma_1_invsqrt) * (W_1.T @ K_2 @ W_1)
    xi, X = np.linalg.eigh(LHS)

    # Increase maximum allowed value slightly to avoid unwanted filtering
    if eta is not None:
        max_val = (1 + eta) / (n * sigma * np.sqrt(2 * np.pi))
    else:  # eta == None
        max_val = float("inf")  # No filtering is applied

    # Filter out unrealistic values
    idx = np.where(np.logical_and(0 <= xi, xi <= max_val))[0].flatten()
    xi = xi[idx]
    X = X[:, idx]

    # Form helper matrix
    D = W_1 @ np.diag(gamma_1_invsqrt) @ X

    return xi, D


def generalied_eigenproblem_direct(K_2, K_1, n, sigma=1.0, eta=1e-3):
    """
    Directly solve generalized eigenvalue problem.

    Parameters
    ----------
    K_2 : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_2 = Z* Z = W* P^2 W.
    K_1 : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_1 = W* Z = W* P W.
    n : int > 0
        Size of original problem.
    sigma : int or float > 0
        Smearing parameter.
    eta : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma. 

    Returns
    -------
    xi : np.ndarray
        The generalized eigenvalues.
    """
    xi = sp.linalg.eig(K_2, K_1, left=False, right=False)

    # Increase maximum allowed value slightly to avoid unwanted filtering
    if eta:
        max_val = (1 + eta) / (n * sigma * np.sqrt(2 * np.pi))
    else:  # eta == None
        max_val = float("inf")  # No filtering is applied

    # Filter out unrealistic values
    idx = np.where(np.logical_and(0 <= xi, xi <= max_val))[0].flatten()
    xi = xi[idx]
    X = X[:, idx]

    return xi


def generalized_eigenproblem_pinv(K_2, K_1, zeta=1e-7):
    """
    Compute the pseudo-inverse: pinv(K_1) * K_2.

    Parameters
    ----------
    K_2 : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_2 = Z* Z = W* P^2 W.
    K_1 : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_1 = W* Z = W* P W.
    zeta : int or float in (0, 1]
        Truncation parameter.

    Returns
    -------
    xi : np.ndarray
        The generalized eigenvalues.
    """

    Xi = np.linalg.pinv(K_1, rcond=zeta) @ K_2
    xi = np.diag(Xi)

    return xi


# --- Unused implementations ---


def _generalized_eigenproblem_kernelunion(K_2, K_1, n, sigma=1.0, zeta=1e-7, eta=1e-3):
    """
    Solve the generalized eigenvalue problem for the spectrum sweeping method.

    Parameters
    ----------
    K_2 : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_2 = Z* Z = W* P^2 W.
    K_1 : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_1 = W* Z = W* P W.
    sigma : int or float > 0
        Smearing parameter.
    zeta : int or float in (0, 1]
        Truncation parameter.
    eta : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma. 

    Returns
    -------
    xi_tilde : np.ndarray
        The generalized eigenvalues.
    C_tilde : np.ndarray
        The generalized eigenvectors.

    References
    ----------
    [1] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 4.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    s_Z, U_Z = np.linalg.eigh(K_2)

    idx = np.where(s_Z > zeta * np.max(s_Z))[0].flatten()
    U_Z_tilde = U_Z[:, idx]

    s_W, U_W = np.linalg.eigh(U_Z_tilde.T @ K_1 @ U_Z_tilde)

    idx = np.where(s_W > zeta * np.max(s_W))[0].flatten()
    s_W_tilde = s_W[idx] ** (-0.5)
    U_W_tilde = U_W[:, idx]

    A = np.outer(s_W_tilde, s_W_tilde) * (U_W_tilde.T @ U_Z_tilde.T @ K_2 @ U_Z_tilde @ U_W_tilde)

    xi, X = np.linalg.eigh(A)

    # Increase maximum allowed value slightly to avoid unwanted filtering
    max_val = (1 + eta) / (n * np.sqrt(2 * np.pi * sigma**2))

    idx_tilde = np.where(np.logical_and(0 <= xi, xi <= max_val))[0].flatten()
    xi_tilde = xi[idx_tilde]
    C_tilde = U_Z_tilde @ U_W_tilde @ np.diag(s_W_tilde) @ X[:, idx_tilde]

    return xi_tilde, C_tilde


def _generalized_eigenproblem_dggev(K_2, K_1, n, sigma=1.0, zeta=1e-7, eta=1e-3):
    """
    Solve the generalized eigenvalue problem for the spectrum sweeping method.

    Parameters
    ----------
    K_2 : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_2 = Z* Z = W* P^2 W.
    K_1 : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_1 = W* Z = W* P W.
    sigma : int or float > 0
        Smearing parameter.
    zeta : int or float in (0, 1]
        Truncation parameter.
    eta : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma. 

    Returns
    -------
    xi_tilde : np.ndarray
        The generalized eigenvalues.
    C_tilde : np.ndarray
        The generalized eigenvectors.

    References
    ----------
    [1] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 4.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """

    alphar, alphai, beta, _, _, _, _ = sp.linalg.lapack.dggev(K_2, K_1)
    idx = np.abs(beta) > 1e-7
    xi_tilde = alphar[idx] / beta[idx]

    return xi_tilde, None


def _generalized_eigenproblem_lstsq(K_2, K_1, n, sigma=1.0, zeta=1e-7, eta=1e-3):
    """
    Solve the generalized eigenvalue problem for the spectrum sweeping method.

    Parameters
    ----------
    K_2 : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_2 = Z* Z = W* P^2 W.
    K_1 : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_1 = W* Z = W* P W.
    sigma : int or float > 0
        Smearing parameter.
    zeta : int or float in (0, 1]
        Truncation parameter.
    eta : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma. 

    Returns
    -------
    xi_tilde : np.ndarray
        The generalized eigenvalues.
    C_tilde : np.ndarray
        The generalized eigenvectors.

    References
    ----------
    [1] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 4.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """

    Xi = np.linalg.lstsq((K_1 + K_1.T) / 2, (K_2 + K_2.T) / 2, rcond=zeta)[0]

    return np.diag(Xi), None


def _generalized_eigenproblem_cholesky(K_2, K_1, n, sigma=1.0, zeta=1e-7, eta=1e-3):
    """
    Solve the generalized eigenvalue problem for the spectrum sweeping method.

    Parameters
    ----------
    K_2 : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_2 = Z* Z = W* P^2 W.
    K_1 : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_1 = W* Z = W* P W.
    sigma : int or float > 0
        Smearing parameter.
    zeta : int or float in (0, 1]
        Truncation parameter.
    eta : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma. 

    Returns
    -------
    xi_tilde : np.ndarray
        The generalized eigenvalues.
    C_tilde : np.ndarray
        The generalized eigenvectors.

    References
    ----------
    [1] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 4.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """

    C = np.linalg.cholesky(K_1)

    C_tilde = np.linalg.pinv(C, rcond=zeta)

    xi_tilde = np.diag(C_tilde @ C_tilde.T @ K_1)

    return xi_tilde, C_tilde
