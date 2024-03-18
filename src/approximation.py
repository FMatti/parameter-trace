"""
Approximation
-------------

Methods for low-rank approximation of matrices.
"""

import numpy as np
import scipy as sp


# --- Unused implementations ---


def _generalized_nystrom_pinv(AX, AY, Y):
    """
    https://arxiv.org/pdf/2010.09649.pdf (Algorithm 2)
    """
    A_nys = AX @ np.linalg.pinv(Y.T @ AX) @ AY.T
    return A_nys


def _generalized_nystrom_qr(AX, AY, Y):
    """
    https://arxiv.org/pdf/2009.11392.pdf (Algorithm 2.1)
    """
    Q, R = np.linalg.qr(Y.T @ AX)
    A_nys = sp.linalg.solve_triangular(R.T, AX.T, lower=True).T @ (Q.T @ AY.T)
    return A_nys


def _generalized_nystrom_stable_qr(AX, AY, Y):
    """
    https://arxiv.org/pdf/2009.11392.pdf (Algorithm 2.1)
    """
    Q, R = np.linalg.qr(Y.T @ AX)
    A_nys = (AX @ np.linalg.pinv(R, rcond=1e-16)) @ (Q.T @ AY.T)
    return A_nys


def _generalized_nystrom_stable_qr_alt(AX, AY, Y):
    """
    https://arxiv.org/pdf/2009.11392.pdf (Section 5.1.1)
    """
    eps = 1e-10
    Q, R, P = sp.linalg.qr(Y.T @ AX, pivoting=True, mode="economic")
    idx = np.abs(np.diag(R)) > eps
    R_1 = R[np.ix_(idx, idx)]
    Q_1 = Q[:, idx]
    Q_2, R_2 = sp.linalg.qr(R_1.T, mode="economic")
    A_nys = (AX @ (Q_2 @ np.linalg.inv(R_2) @ Q_1)) @ (Q.T @ AY.T)
    return A_nys


def _nystrom_svd(AX, X):
    """
    https://arxiv.org/pdf/2306.12418.pdf (Algorithm 5.6)
    """
    # Apply an epsilon-shift
    eps = 1e-16
    Y = AX + eps*X

    # Cholesky decomposition and subsequent processing
    C = np.linalg.cholesky(X.T @ Y)
    Z = sp.linalg.solve_triangular(C.T, Y.T).T
    U, S, _ = np.linalg.svd(Z, full_matrices=False)

    # Readjustment for epsilon-shift
    Xi = np.maximum(0, S**2 - eps)

    return Xi, U
