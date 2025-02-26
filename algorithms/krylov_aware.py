"""
krylov_aware.py
---------------

Implementation of Krylov-aware trace estimator.
"""

import numpy as np
import scipy as sp

from .helpers import gaussian_kernel

def lanczos(A, X, k, reorth_tol=0.7, return_matrix=True, extend_matrix=True, dtype=None):
    """
    Implements the Krylov-decomposition of a Hermitian matrix or linear operator
    A with the Lanczos method [1]. The decomposition consists of an orthogonal
    matrix U and a Hermitian tridiagonal matrix T which satisfy

        A @ U[:, :k] = U[:, :k+1] @ T

    after k iterations of the Lanczos method. Multiple starting vectors can be
    passed as X = [x_1, x_2, ..., x_m].

    Parameters
    ----------
    A : np.ndarray (n, n)
        Symmetric matrix.
    X : np.ndarray (n, ) or (n, m,^)
        Collection of starting vectors.
    k : int > 0
        Number of iterations.
    reorth_tol : float < 1
        The tolerance for reorthogonalizing the Krylov basis between iterations.
    return_matrix : bool
        Whether to return the (full) tridiagonal matrix H or arrays of its
        diagonal and off-diagonal elements.
    extend_matrix : bool
        Whether to extend the orthogonal matrix U with one more column after
        the last iteration.
    dtype : type or str
        The datatype of the matrix, such that auxiliary arrays are the same.

    Returns
    -------
    U : np.ndarray (m, n, k + 1)
        Orthogonal basis of the block Krylov subspace(s) K(A, X).
    T or a, b : np.ndarray (m, k + 1, k + 1) or (m, k + 1), (m, k + 1)
        The block tridiagonal matrix T or its tridiagonal blocks a, b.

    References
    ----------
    [1] Lanczos, C. (1950). "An iteration method for the solution of the
        eigenvalue problem of linear differential and integral operators".
        Journal of Research of the National Bureau of Standards. 45 (4): 255–282.
        doi:10.6028/jres.045.026.
    """

    # Pre-process the starting vector/matrix
    if X.ndim < 2:
        X = X[:, np.newaxis]
    n, m = X.shape

    # Determine the range of A in case it is not an array/matrix
    dtype = A.dtype if dtype is None else dtype

    # Initialize arrays for storing the block-tridiagonal elements
    a = np.empty((k, m), dtype=dtype)
    b = np.empty((k + 1, m), dtype=dtype)
    U = np.empty((k + 1, n, m), dtype=dtype)

    # Specify initial iterate
    b[0] = np.linalg.norm(X, axis=0)
    U[0] = X / b[0]

    # Perform k iterations of the Lanczos algorithm
    for j in range(k):

        # Generate and orthogonalize next element(s) in Krylov-basis
        w = A @ U[j]
        a[j] = np.sum(U[j].conj() * w, axis=0)
        u_tilde = w - U[j] * a[j] - (U[j-1] * b[j] if j > 0 else 0)

        # Reorthogonalize any columns which might have lost orthogonality
        idx, = np.where(np.linalg.norm(u_tilde, axis=0) < reorth_tol * np.linalg.norm(w, axis=0))
        if len(idx) >= 1:
            h_hat = np.einsum("ijk,jk->ik", U[:j + 1, :, idx].conj(), u_tilde[:, idx])
            a[j, idx] += h_hat[-1]
            if j > 0:
                b[j - 1, idx] += h_hat[-2]
            u_tilde[:, idx] -= np.einsum("ijk,ik->jk", U[:j + 1, :, idx].conj(), h_hat)

        b[j + 1] = np.linalg.norm(u_tilde, axis=0)
        U[j + 1] = u_tilde / b[j + 1]

    U = np.einsum("ijk->kji", U)
    a = np.einsum("ij->ji", a)
    b = np.einsum("ij->ji", b)

    if return_matrix:
        # Assemble block-tridiagonal matrices T[i] for i = 1, ..., m
        T = np.zeros((m, k + 1, k), dtype=a.dtype)
        T[:, np.arange(k), np.arange(k)] = a
        T[:, np.arange(1, k + 1), np.arange(k)] = b[:, 1:]
        T[:, np.arange(k - 1), np.arange(1, k)] = b[:, 1:-1]
        if not extend_matrix:
            T = T[:, :k, :]
        return U, T

    return U, a, b


def block_lanczos(A, X, k, reorth_steps=-1, return_matrix=True, extend_matrix=True, dtype=None):
    """
    Implements the Krylov-decomposition of a Hermitian matrix or linear operator
    A with the block Lanczos method [2]. The decomposition consists of an
    orthogonal matrix U and a Hermitian tridiagonal matrix T which satisfy

        A @ U[:, :k * n_X] = U[:, :(k + 1) * n_X] @ T

    after k iterations of the Lanczos method.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Symmetric matrix with eigenvalues between (-1, 1).
    X : np.ndarray (n, m)
        Starting block.
    k : int > 0
        Number of iterations.
    reorth_steps : int <= k
        The number of iterations in which to reorthogonalize. To always
        reorthogonalize, use -1.
    return_matrix : bool
        Whether to return the (full) tridiagonal matrix H or arrays of its
        diagonal and off-diagonal elements.
    extend_matrix : bool
        Whether to extend the orthogonal matrix U with one more column after
        the last iteration.
    dtype : type or str
        The datatype of the matrix, such that auxiliary arrays are the same.

    Returns
    -------
    U : np.ndarray (n, (k + 1) * m)
        Orthogonal basis of the block Krylov subspace K(A, X).
    T or a, b : np.ndarray ((k + 1) * m, (k + 1) * m)
        The block tridiagonal matrix T or its tridiagonal blocks a, b.
    
    References
    ----------
    [2] Montgomery, P. L. (1995). "A Block Lanczos Algorithm for Finding
        Dependencies over GF(2)". Lecture Notes in Computer Science. EUROCRYPT.
        Vol. 921. Springer-Verlag. pp. 106–120. doi:10.1007/3-540-49264-X_9.
    [3] Chen T. and Hallman, E. (2023). "Krylov-Aware Stochastic Trace
        Estimation". SIAM Journal on Matrix Analysis and Applications.
        Vol. 44 (3). doi:10.1137/22M1494257.
    """

    # Pre-process the starting vector/matrix
    if X.ndim < 2:
        X = X[:, np.newaxis]
    n, m = X.shape

    # Determine datatype of A
    dtype = A.dtype if dtype is None else dtype

    # Initialize arrays for storing the block-tridiagonal elements
    a = np.empty((k, m, m), dtype=dtype)
    b = np.empty((k + 1, m, m), dtype=dtype)
    U = np.empty((k + 1, n, m), dtype=dtype)

    # Specify initial iterate
    U[0], b[0] = np.linalg.qr(X)

    # Perform k iterations of the Lanczos algorithm
    for j in range(k):

        w = A @ U[j]
        a[j] = U[j].conj().T @ w
        u_tilde = w - U[j] @ a[j] - (U[j-1] @ b[j].conj().T if j > 0 else 0)

        # reorthogonalization
        if j > 0 and (j < reorth_steps or reorth_steps == -1):
            h_hat = np.swapaxes(U[:j], 0, 1).reshape(n, -1).conj().T @ u_tilde
            a[j] += h_hat[-1]
            u_tilde = u_tilde - np.swapaxes(U[:j], 0, 1).reshape(n, -1) @ h_hat

        # Pivoted QR
        z_tilde, R, p = sp.linalg.qr(u_tilde, pivoting=True, mode="economic")
        b[j + 1] = R[:, np.argsort(p)]

        # Orthogonalize again if R is rank deficient
        if reorth_steps > j or reorth_steps == -1:
            r = np.abs(np.diag(b[j + 1]))
            r_idx = np.nonzero(r < np.max(r) * 1e-10)[0]
            z_tilde[:, r_idx] = z_tilde[:, r_idx] - U[j] @ (U[j].conj().T @ z_tilde[:, r_idx])
            z_tilde[:, r_idx], R = np.linalg.qr(z_tilde[:, r_idx])
            z_tilde[:, r_idx] *= np.sign(np.diag(R))
            U[j + 1] = z_tilde

    U = np.einsum("ijk->jik", U).reshape(n, -1)

    if return_matrix:

        x, y = np.meshgrid(np.arange(m), np.arange(m))
        idx = np.add.outer(m * np.arange(k), x).ravel()
        idy = np.add.outer(m * np.arange(k), y).ravel()
        if extend_matrix:
            row = np.concatenate((idy, idy + m, idy[:-m ** 2]))
            col = np.concatenate((idx, idx, idx[:-m ** 2] + m))
            data = np.concatenate((a.ravel(), b[1:k+1].ravel(), np.einsum("ijk->ikj", b[1:k].conj()).ravel()))
            T = sp.sparse.coo_matrix((data, (row, col)), shape=((k + 1)*m, k*m))
        else:
            row = np.concatenate((idy, idy[:-m ** 2] + m, idy[:-m ** 2]))
            col = np.concatenate((idx, idx[:-m ** 2], idx[:-m ** 2] + m))
            data = np.concatenate((a.ravel(), b[1:k].ravel(), np.einsum("ijk->ikj", b[1:k].conj()).ravel()))
            T = sp.sparse.coo_matrix((data, (row, col)), shape=(k*m, k*m))
        return U, T
    return U, a, b


def krylov_aware(A, t=0, n_iter=10, n_reorth=5, n_Omega=10, n_Psi=10, f=gaussian_kernel):
    """
    Krylov-aware stochastic trace estimator to approximate the trace of a 
    (parametrized) matrix function Tr(f_t(A)).

    Parameters
    ----------
    A : np.ndarray (n, n)
        Symmetric matrix with eigenvalues between (-1, 1).
    t : int, float, or np.ndarray (n_t)
        Parameter values at which estimator is evaluated.
    n_iter : int > 0
        Number of block Lanczos and Lanczos iterations.
    reorth_steps : int <= n_iter
        The number of iterations of block Lanczos in which to reorthogonalize.
        To always reorthogonalize, use -1.
    n_Omega : int > 0
        Size of sketching matrix in the block Lanczos low-rank approximation.
    n_Psi : int > 0
        Number of queries with Girard-Hutchinson estimator.
    f : callable or None
        Function of f(t, x) of a parameter t and scalar x: f_t(A) = f(t, A).
        If None is specified, f is the identity.
        
    Returns
    -------
    trace : np.ndarray (n_t, )
        The approximations of Tr(f_t(A)) at all values of the parameter t.
    
    References
    ----------
    [3] Chen T. and Hallman, E. (2023). "Krylov-Aware Stochastic Trace
        Estimation". SIAM Journal on Matrix Analysis and Applications.
        Vol. 44 (3). doi:10.1137/22M1494257.
    """
    # Compute block Lanczos low-rank approximation trace quadrature of A
    if n_Omega > 0:
        Omega = np.random.randn(A.shape[0], n_Omega)
        Q, T = block_lanczos(A, Omega, n_iter, extend_matrix=False, reorth_steps=n_reorth)
        nodes, T_evec = np.linalg.eigh(T.toarray())
        weights = np.linalg.norm(T_evec[:(n_reorth + 1) * n_Omega], axis=0)**2
    else:
        nodes = []
        weights = []

    # Transform the Girard-Hutchinson query-vectors
    Psi = np.random.randn(A.shape[0], n_Psi)
    if n_Omega > 0:
        Psi -= Q[:, :(n_reorth + 1) * n_Omega] @ (Q[:, :(n_reorth + 1) * n_Omega].T @ Psi)

    # Approximate the Girard-Hutchinson estimator of the approximation residual
    _, a_rem, b_rem = lanczos(A, Psi, n_iter, extend_matrix=False, return_matrix=False, reorth_tol=0)
    for a, b in zip(a_rem, b_rem):
        nodes_, T_evec = sp.linalg.eigh_tridiagonal(a, b[1:-1])
        weights_ = T_evec[0]**2 * (A.shape[0] - (n_reorth + 1) * n_Omega) / n_Psi
        nodes = np.append(nodes, nodes_)
        weights = np.append(weights, weights_)

    # Define the function f as the identity, if none is specified
    if f is None:
        fun = lambda t, x: x
    else:
        fun = lambda t, x: f(t, x)

    # Evaluate the quadrature
    return fun(t, nodes) @ weights
