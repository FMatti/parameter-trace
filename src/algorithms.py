"""
Algorithms
----------

Implementation of the algorithms for spectral density estimation.
"""

import sys

import numpy as np
import scipy as sp

from src.kernel import gaussian_kernel
from src.interpolation import chebyshev_coefficients, exponentiate_chebyshev_coefficients_cosine_transform, chebyshev_recurrence
from src.eigenproblem import generalized_eigenproblem_standard, generalized_eigenproblem_pinv, generalied_eigenproblem_direct
from src.utils import continued_fraction


# Increase recursion limit for Haydock's method
sys.setrecursionlimit(5000)


def DGC(A, t, m, sigma, n_v, kernel=gaussian_kernel, rho=0, seed=0, nonnegative=False):
    """
    Delta-Gauss-Chebyshev method for estimating the spectral density.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Symmetric matrix with eigenvalues between (-1, 1).
    t : np.ndarray (n_t,)
        A set of points at which the DOS is to be evaluated.
    m : int > 0
        Degree of the polynomial.
    sigma : int or float > 0
        Smearing parameter.
    n_v : int > 0
        Number of Hutchinson's queries.
    kernel : function
        The smoothing kernel applied to the spectral density.
    rho : int or float > 0
        The shift which is applied to the kernel before computation.
    seed : int >= 0
        The seed for generating the random matrix W.
    nonnegative : bool
        Force interpolant to be non-negative (for non-negative functions).
        
    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [1] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 2.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    n = A.shape[0]

    # Polynomial expansion
    g = lambda x: kernel(x, n=n, sigma=sigma) + rho
    mu = chebyshev_coefficients(t, m, function=g, nonnegative=nonnegative)

    # Compute recurrence from Chebyshev expansion
    Psi = np.random.randn(n, n_v)
    phi_tilde = chebyshev_recurrence(mu, A, T_0=Psi, L=lambda x: np.multiply(Psi, x).sum() / n_v)

    return phi_tilde - rho * np.trace(Psi.T @ Psi) / n_v


def NC(A, t, m, sigma, n_v, k=1, zeta=1e-7, kappa=1e-5, eta=1e-3, kernel=gaussian_kernel, square_coefficients="transformation", eigenproblem="standard", rho=0, seed=0, nonnegative=False):
    """
    Nyström-Chebyshev method for estimating the spectral density.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Symmetric matrix with eigenvalues between (-1, 1).
    t : np.ndarray (n_t,)
        A set of points at which the DOS is to be evaluated.
    m : int > 0
        Degree of the Chebyshev polynomial.
    sigma : int or float > 0
        Smearing parameter.
    n_v : int > 0
        Size of sketching matrix (in Nyström approximation).
    k : int > 0
        The approximation method used (1 = Nyström, 2 = RSVD, 3 = SI-Nyström)
    zeta : int or float in (0, 1]
        Truncation parameter for enforcing conditioning of eigenvalue problem.
    kappa : float > 0
        The threshold on the Hutchinson estimate of g_sigma. If it is below this
        value, instead of solving the possibly ill-conditioned generalized
        eigenvalue problem, we set the spectral density at that point to zero.
    eta : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma.
    kernel : function
        The smoothing kernel applied to the spectral density.
    square_coefficients : str or None
        Method by which the coefficients of the squared Gaussian are computed.
         -> transformation = Compute coefficients with discrete cosine transform
         -> interpolation = Interpolate the squared function
    eigenproblem : str
        Resolution method of the generalized eigenvalue problem in SS methods.
         -> standard = As proposed in [1] (project out kern(K_1))
         -> direct = Directly solve the generalized eigenproblem
         -> pinv = Directly compute pseudoinverse
    rho : int or float > 0
        The shift which is applied to the kernel before computation.
    seed : int >= 0
        The seed for generating the random matrix W.
    nonnegative : bool
        Force interpolant to be non-negative (for non-negative functions).
        
    Returns
    -------
    phi_hat : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [1] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 5.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix
    n = A.shape[0]

    # Convert evaluation point(s) to numpy array
    if not isinstance(t, np.ndarray):
        t = np.array(t).reshape(-1)

    # Compute coefficients of Chebyshev expansion of the smoothing kernel
    g = lambda x: kernel(x, n=n, sigma=sigma) + rho
    if square_coefficients == "interpolation":
        mu = chebyshev_coefficients(t, k * m, function=lambda x: g(x) ** k, nonnegative=nonnegative)
        nu = chebyshev_coefficients(t, (k + 1) * m, function=lambda x: g(x) ** (k + 1),  nonnegative=nonnegative)
    else:  # square_coefficients == "transformation":
        mu_1 = chebyshev_coefficients(t, m, function=g)
        mu = exponentiate_chebyshev_coefficients_cosine_transform(mu_1, k=k)
        nu = exponentiate_chebyshev_coefficients_cosine_transform(mu_1, k=k + 1)

    # Compute recurrence from Chebyshev expansion
    Omega = np.random.randn(n, n_v)
    K_1 = chebyshev_recurrence(mu, A, T_0=Omega, L=lambda x: Omega.T @ x, final_shape=(n_v, n_v))
    K_2 = chebyshev_recurrence(nu, A, T_0=Omega, L=lambda x: Omega.T @ x, final_shape=(n_v, n_v))

    # Trace computation
    phi_hat = np.empty(t.shape[0])
    for i in range(t.shape[0]):
        # Check if rank of if Hutchinson (k=1) for Tr(g^{(m)}(tI-A)) is almost zero
        if np.trace(K_1[i]) / n_v < kappa:
            phi_hat[i] = 0
            continue
        else:
            if eigenproblem == "pinv":
                Xi = generalized_eigenproblem_pinv(K_2[i], K_1[i], zeta=zeta)
            elif eigenproblem == "direct":
                Xi = generalied_eigenproblem_direct(K_2[i], K_1[i], n=n, sigma=sigma, eta=eta)
            else:  # eigenproblem == "standard":
                Xi = generalized_eigenproblem_standard(K_2[i], K_1[i], n=n, sigma=sigma, zeta=zeta, eta=eta)[0]
            phi_hat[i] = np.sum(Xi)

    return phi_hat - rho * n_v


def NCPP(A, t, m, sigma, n_v, n_v_tilde=None, k=1, zeta=1e-7, kappa=1e-5, eta=1e-3, kernel=gaussian_kernel, square_coefficients="transformation", rho=0, seed=0, nonnegative=False):
    """
    Nyström-Chebyshev++ method for estimating the spectral density.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Symmetric matrix with eigenvalues between (-1, 1).
    t : np.ndarray (n_t,)
        A set of points at which the DOS is to be evaluated.
    m : int > 0
        Degree of Chebyshev the polynomial.
    sigma : int or float > 0
        Smearing parameter.
    n_v : int > 0
        Size of sketching matrix (in Nyström approximation).
    n_v_tilde : int > 0
        Number of Hutchinson's queries.
    k : int > 0
        The approximation method used (1 = Nyström, 2 = RSVD, 3 = SI-Nyström)
    zeta : int or float in (0, 1]
        Truncation parameter.
    kappa : float > 0
        The threshold on the Hutchinson estimate of g_sigma. If it is below this
        value, instead of solving the possibly ill-conditioned generalized
        eigenvalue problem, we set the spectral density at that point to zero.
    eta : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma.
    kernel : function
        The smoothing kernel applied to the spectral density.
    square_coefficients : str or None
        Method by which the coefficients of the squared Gaussian are computed.
         -> transformation = Compute coefficients with discrete cosine transform
         -> interpolation = Interpolate the squared function
    rho : int or float > 0
        The shift which is applied to the kernel before computation.
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    phi_breve : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [1] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 7.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    n = A.shape[0]

    # Convert evaluation point(s) to numpy array
    if not isinstance(t, np.ndarray):
        t = np.array(t).reshape(-1)

    # Preprocess the number of random vectors
    if n_v == 0:
        return DGC(A, t, m, sigma, n_v_tilde, kernel, seed)
    if n_v_tilde is None:  # Evenly distribute mat-vecs
        n_v_tilde = n_v // 2
        n_v = n_v // 2
    elif n_v_tilde == 0:
        return NC(A, t, m, sigma, n_v, k, zeta, kappa, eta, kernel, square_coefficients, seed=seed)

    # Compute coefficients of Chebyshev expansion of the smoothing kernel
    g = lambda x: kernel(x, n=n, sigma=sigma) + rho
    if square_coefficients == "transformation":
        mu_1 = chebyshev_coefficients(t, m, function=g, nonnegative=nonnegative)
        mu = exponentiate_chebyshev_coefficients_cosine_transform(mu_1, k=k)
        nu = exponentiate_chebyshev_coefficients_cosine_transform(mu_1, k=k + 1)
        mu_L = mu_1 if k < 3 else exponentiate_chebyshev_coefficients_cosine_transform(mu_1, k=(k + 1) // 2)
        nu_L = mu_L if k % 2 == 1 else exponentiate_chebyshev_coefficients_cosine_transform(mu_1, k=(k + 2) // 2)
    else:  # square_coefficients == "interpolation":
        mu_1 = chebyshev_coefficients(t, m, function=g, nonnegative=nonnegative)
        mu = chebyshev_coefficients(t, k * m, function=lambda x: g(x) ** k, nonnegative=nonnegative)
        nu = chebyshev_coefficients(t, k * m, function=lambda x: g(x) ** (k + 1), nonnegative=nonnegative)
        mu_L = mu_1 if k < 3 else chebyshev_coefficients(t, k * m, function=lambda x: g(x) ** ((k + 1) // 2))
        nu_L = mu_L if k % 2 == 1 else chebyshev_coefficients(t, k * m, function=lambda x: g(x) ** ((k + 2) // 2))

    # Compute recurrence from Chebyshev expansion
    Omega = np.random.randn(n, n_v)
    Psi = np.random.randn(n, n_v_tilde)
    K_1 = chebyshev_recurrence(mu, A, T_0=Omega, L=lambda x: Omega.T @ x, final_shape=(n_v, n_v))
    K_2 = chebyshev_recurrence(nu, A, T_0=Omega, L=lambda x: Omega.T @ x, final_shape=(n_v, n_v))
    L_1 = chebyshev_recurrence(mu_L, A, T_0=Omega, L=lambda x: Psi.T @ x, final_shape=(n_v_tilde, n_v))
    L_2 = L_1 if k % 2 == 1 else chebyshev_recurrence(nu_L, A, T_0=Omega, L=lambda x: Psi.T @ x, final_shape=(n_v_tilde, n_v))
    ell = chebyshev_recurrence(mu_1, A, T_0=Psi, L=lambda x: np.sum(np.multiply(Psi, x)), final_shape=())

    phi_breve = np.zeros(t.shape[0])
    for i in range(t.shape[0]):
        if np.trace(K_1[i]) / n_v < kappa:  # Hutchinson for Tr(g^{(m)}(tI-A))
            continue
        xi_tilde, C_tilde = generalized_eigenproblem_standard(K_2[i], K_1[i], n=n, sigma=sigma, zeta=zeta, eta=eta)
        T = np.trace(L_1[i] @ C_tilde @ C_tilde.conjugate().T @ L_2[i].T)
        phi_breve[i] = np.sum(xi_tilde) + (ell[i] - T) / n_v_tilde

    return phi_breve - rho * n


def Lanczos(A, x, k, reorth_tol=0.7):
    """
    Compute coefficients of symmetric tridiagonal matrix T from Lanczos.

        T = np.diagonal([b, a, b], [-1, 0, 1])

    Parameters
    ----------
    A : np.ndarray (n, n)
        Symmetric matrix.
    x : np.ndarray (n,)
        Starting vector for Lanczos method.
    k : int > 0
        Number of Lanczos iterations.

    Returns
    -------
    a : np.ndarray (k,)
        The diagonal elements of the tridiagonal matrix from Lanczos.
    b : np.ndarray (k,)
        The secondary-diagonal elements of the tridiagonal matrix from Lanczos.

    References
    ----------
    [2] Lanczos, C. An Iteration Method for the Solution of the Eigenvalue
        Problem of Linear Differential and Integral Operators. Journal of
        Research of the National Bureau of Standards. 45, 255-282 (1950)},
        DOI: https://doi.org/10.6028/jres.045.026
    """
    # Initialize arrays for storing the diagonal and secondary-diagonal elements
    a = np.empty(k)
    b = np.empty(k)

    # Orthogonal matrix which is constructed in Lanczos algorithm
    U = np.empty((A.shape[0], k + 1))
    U[:, 0] = x / np.linalg.norm(x)

    # Lanczos iterations
    for j in range(k):
        w = A @ U[:, j]
        a[j] = U[:, j].T @ w
        u_tilde = w -  U[:, j] * a[j] - (U[:, j - 1] * b[j - 1] if j > 0 else 0) 

        # Perform reorthogonalization
        if np.linalg.norm(u_tilde) <= reorth_tol * np.linalg.norm(w):
            # Twice is enough
            h_hat = U[:, : j + 1].T @ u_tilde
            a[j] += h_hat[-1]
            if j > 0:
                b[j - 1] += h_hat[-2]
            u_tilde -= U[:, : j + 1] @ h_hat

        b[j] = np.linalg.norm(u_tilde)
        U[:, j + 1] = u_tilde / b[j]

    return a, b


def Haydock(A, t, m, sigma, n_v, seed=0, kernel=None, eta=None):
    """
    Haydock's method.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Symmetric matrix with eigenvalues between (-1, 1).
    t : np.ndarray (n_t,)
        A set of points at which the DOS is to be evaluated.
    m : int > 0
        The number of Lanczos iterations.
    sigma : int or float > 0
        Smearing parameter.
    n_v : int > 0
        Number of random vectors used in Monte-Carlo estimate.
    seed : int >= 0
        The seed for generating the random matrix W.
    kernel : None
        Unused dummy argument for compatibility reasons.
    eta : None
        Unused dummy argument for compatibility reasons.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [3] L. Lin, Y. Saad, C. Yang. Approximating Spectral Densities of Large Matrices.
        SIAM Reviev 58(1) (2016). Section 3.2.2. 
        Link: https://doi.org/10.1137/130934283
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    n = A.shape[0]

    phi_tilde = np.zeros(len(t))
    for _ in range(n_v):
        # Compute tridiagonal matrix from Lanczos for random vector
        v = np.random.randn(n)
        a, b = Lanczos(A, v, m)
        phi_tilde += np.imag(continued_fraction((t + 1j*sigma), a, b))

    phi_tilde *= - 1 / (n_v * np.pi)
    return phi_tilde


# --- Unused implementations ---


from src.approximation import _generalized_nystrom_pinv, _generalized_nystrom_qr, _generalized_nystrom_stable_qr


def _KPM(A, t, m, n_v, seed=0, sigma=None):
    """
    Kernel polynomial method for computing the spectral density.

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        Symmetric matrix with eigenvalues between (-1, 1).
    t : np.ndarray of shape (n_t,)
        A set of points at which the DOS is to be evaluated.
    m : int > 0
        Degree of the polynomial.
    n_v : int > 0
        Number of random vectors.
    seed : int >= 0
        The seed for generating the random matrix W.
    sigma : None
        Unused dummy-argument to match function signature of other algorithms.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [4] Lin, L. Approximating spectral densities of large matrices: old and new.
        Math/CS Seminar, Emory University (2015).
        Link: https://math.berkeley.edu/~linlin/presentations/201753_DOS.pdf
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    n = A.shape[0]

    # Initializations
    mu = np.zeros(m + 1)
    W = np.random.randn(n, n_v)
    V_c = W.copy()
    V_m = np.zeros((n, n_v))
    V_p = np.zeros((n, n_v))

    # Chebyshev recursion
    for l in range(m + 1):
        mu[l] = np.sum(np.multiply(W, V_c)) / (n_v * n * np.pi)
        V_p = (1 if l == 0 else 2) * A @ V_c - V_m
        V_m = V_c.copy()
        V_c = V_p.copy()

    # Computation of the approximate spectral density
    phi_tilde = 1 / np.sqrt(1 + t**2) * np.polynomial.chebyshev.Chebyshev(mu)(t)
    return phi_tilde


def _SLQ(A, t, sigma, n_v, m=200, seed=0):
    """
    Stochastic Lanczos Quadrature.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Symmetric matrix with eigenvalues between (-1, 1).
    t : np.ndarray (n_t,)
        A set of points at which the DOS is to be evaluated.
    sigma : int or float > 0
        Smearing parameter.
    n_v : int > 0
        The number of Lanczos iterations.
    m : int > 0
        Number of random vectors used in Monte-Carlo estimate.
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [5] T. Chen, T. Trogdon, S. Ubaru. Analysis of stochastic Lanczos quadrature
        for spectrum approximation. PMLR 139:1728-1739 (2021).
        Link: http://proceedings.mlr.press/v139/chen21s/chen21s.pdf
    """

    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    n = A.shape[0]

    phi_tilde = np.zeros_like(t)
    for _ in range(m):
        x = np.random.randn(n)
        a, b = Lanczos(A, x / np.linalg.norm(x), n_v)
        theta, S = sp.linalg.eigh_tridiagonal(a[: n_v], b[: n_v - 1])
        t_minus_theta = np.subtract.outer(t, theta)
        phi_tilde += gaussian_kernel(t_minus_theta, sigma=sigma) @ S[0]**2

    return phi_tilde / n_v


def _randomized_lowrank_decomposition(A, r, c=10, seed=0):
    """
    Randomized low-rank decomposition of a symmetric matrix. Format: A = ZBZ*

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        Symmetric matrix which will be approximated.
    r : int > 0
        Approximate rank of the matrix A.
    c : int > 0
        Small constant by which the random matrix will be larger than r.
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    Z : np.ndarray of shape (n, r + c)
        Basis spanning the space of the low-rank approximation.
    B : np.ndarray of shape (r + c, r + c)
        Approximate decomposition of the matrix P in the basis Z.

    References
    ----------
    [1] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 3.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    n = A.shape[0]

    # Compute randomized low-rank approximation
    W = np.random.randn(n, r + c)
    Z = A @ W
    B = np.linalg.pinv(W.T @ Z, hermitian=True)

    return Z, B


def _randomized_trace_estimation(A, n_v, n_v_tilde):
    """
    Robust and efficient method for estimating the trace of a low-rank matrix.

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        Symmetric matrix which will be approximated.
    n_v : int > 0
        Number of randomized vectors in random matrix.
    n_v_tilde : int > 0
        Number of randomized vectors in other random matrix.

    Returns
    -------
    trace : float
        Approximation of the trace of the matrix P.

    References
    ----------
    [1] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 6.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Determine size of matrix i.e. number of eigenvalues 
    n = A.shape[0]

    W = np.random.randn(n, n_v)
    W_tilde = np.random.randn(n, n_v_tilde)
    K_1 = W.T @ (A @ W)
    K_2 = W.T @ (np.linalg.matrix_power(A, 2) @ W)
    K_C = W_tilde.T @ (A @ W)
    K_1_tilde = W_tilde.T @ (A @ W_tilde)

    xi_tilde, C_tilde = generalized_eigenproblem_standard(K_2, K_1, n=n)

    T = K_C @ C_tilde @ C_tilde.T @ K_C.T
    trace = np.sum(xi_tilde) + (np.trace(K_1_tilde - T)) / n_v_tilde

    return trace


def _NyChebSI(A, t, m, sigma, n_v, zeta=1e-7, eta=1e-3, kernel=gaussian_kernel, eigenproblem="standard", seed=0):
    """
    Spectrum sweeping method using the Delta-Gauss-Chebyshev expansion for
    estimating the spectral density.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Symmetric matrix with eigenvalues between (-1, 1).
    t : np.ndarray (n_t,)
        A set of points at which the DOS is to be evaluated.
    m : int > 0
        Degree of the polynomial.
    sigma : int or float > 0
        Smearing parameter.
    n_v : int > 0
        Number of random vectors.
    zeta : int or float in (0, 1]
        Truncation parameter.
    eta : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma.
    kernel : function
        The smoothing kernel applied to the spectral density.
    eigenproblem : str
        Resolution method of the generalized eigenvalue problem in SS methods.
         -> standard = As proposed in [1] (project out kern(K_1))
         -> kernelunion = Project out union of kern(K_1) and kern(K_2)
         -> pinv = Directly compute pseudoinverse
         -> dggev = Use KZ algorithm
         -> lstsq = Solve leastsquares problem
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [1] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 5.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    n = A.shape[0]

    if not isinstance(t, np.ndarray):
        t = np.array(t).reshape(-1)

    # Polynomial expansion
    g = lambda x: kernel(x, n=n, sigma=sigma)
    mu = chebyshev_coefficients(t, m, function=g)
    #nu = squared_chebyshev_coefficients_cosine_transform(mu)

    # Do recurrence
    W = np.random.randn(n, n_v)
    Z = chebyshev_recurrence(mu, A, T_0=W, final_shape=(n, n_v))
    Y = chebyshev_recurrence(mu, A, final_shape=(n, n))

    phi_tilde = np.empty(t.shape[0])
    for i in range(t.shape[0]):
        phi_tilde[i] = np.trace(Z[i] @ np.linalg.pinv(Z[i].T @ Z[i]) @ Z[i].T @ Y[i])
        #if eigenproblem == "kernelunion":
        #    Xi = generalized_eigenproblem_kernelunion(Z[i].T @ Z[i], W.T @ Z[i], n=n, sigma=sigma, zeta=zeta, eta=eta)[0]
        #elif eigenproblem == "pinv":
        #    Xi = generalized_eigenproblem_pinv(Z[i].T @ Z[i], W.T @ Z[i], n=n, sigma=sigma, zeta=zeta, eta=eta)[0]
        #elif eigenproblem == "dggev":
        #    Xi = generalized_eigenproblem_dggev(Z[i].T @ Z[i], W.T @ Z[i], n=n, sigma=sigma, zeta=zeta, eta=eta)[0]
        #elif eigenproblem == "lstsq":
        #    Xi = generalized_eigenproblem_lstsq(Z[i].T @ Z[i], W.T @ Z[i], n=n, sigma=sigma, zeta=zeta, eta=eta)[0]
        #else:
        #    Xi = generalized_eigenproblem_standard(Y[i].T @ Y[i], Z[i].T @ Y[i], n=n, sigma=sigma, zeta=zeta, eta=eta)[0]
        #phi_tilde[i] = np.sum(Xi)

    return phi_tilde


def _NyCheb(A, t, m, sigma, n_v, zeta=1e-7, eta=1e-3, kernel=gaussian_kernel, eigenproblem="standard", seed=0):
    """
    Spectrum sweeping method using the Delta-Gauss-Chebyshev expansion for
    estimating the spectral density.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Symmetric matrix with eigenvalues between (-1, 1).
    t : np.ndarray (n_t,)
        A set of points at which the DOS is to be evaluated.
    m : int > 0
        Degree of the polynomial.
    sigma : int or float > 0
        Smearing parameter.
    n_v : int > 0
        Number of random vectors.
    zeta : int or float in (0, 1]
        Truncation parameter.
    eta : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma.
    kernel : function
        The smoothing kernel applied to the spectral density.
    eigenproblem : str
        Resolution method of the generalized eigenvalue problem in SS methods.
         -> standard = As proposed in [1] (project out kern(K_1))
         -> kernelunion = Project out union of kern(K_1) and kern(K_2)
         -> pinv = Directly compute pseudoinverse
         -> dggev = Use KZ algorithm
         -> lstsq = Solve leastsquares problem
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [1] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 5.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    n = A.shape[0]

    if not isinstance(t, np.ndarray):
        t = np.array(t).reshape(-1)

    # Polynomial expansion
    g = lambda x: kernel(x, n=n, sigma=sigma)
    mu = chebyshev_coefficients(t, m, function=g)

    # Do recurrence
    W = np.random.randn(n, n_v)
    Z = chebyshev_recurrence(mu, A, T_0=W, final_shape=(n, n_v))

    phi_tilde = np.empty(t.shape[0])
    for i in range(t.shape[0]):
        if eigenproblem == "kernelunion":
            Xi = _generalized_eigenproblem_kernelunion(Z[i].T @ Z[i], W.T @ Z[i], n=n, sigma=sigma, zeta=zeta, eta=eta)[0]
        elif eigenproblem == "pinv":
            Xi = generalized_eigenproblem_pinv(Z[i].T @ Z[i], W.T @ Z[i], n=n, sigma=sigma, zeta=zeta, eta=eta)[0]
        elif eigenproblem == "dggev":
            Xi = _generalized_eigenproblem_dggev(Z[i].T @ Z[i], W.T @ Z[i], n=n, sigma=sigma, zeta=zeta, eta=eta)[0]
        elif eigenproblem == "lstsq":
            Xi = _generalized_eigenproblem_lstsq(Z[i].T @ Z[i], W.T @ Z[i], n=n, sigma=sigma, zeta=zeta, eta=eta)[0]
        else:
            Xi = generalized_eigenproblem_standard(Z[i].T @ Z[i], W.T @ Z[i], n=n, sigma=sigma, zeta=zeta, eta=eta)[0]
        phi_tilde[i] = np.sum(Xi)

    return phi_tilde


def _GenNyCheb(A, t, m, sigma, n_v, c1=1/4, c2=1/2, nystrom_version="pinv", seed=0):
    """

    TODO
    
    Parameters
    ----------
    A : np.ndarray (n, n)
        Symmetric matrix with eigenvalues between (-1, 1).
    t : np.ndarray (n_t,)
        A set of points at which the DOS is to be evaluated.
    m : int > 0
        Degree of the polynomial.
    sigma : int or float > 0
        Smearing parameter.
    n_v : int > 0
        Number of random vectors.
    c_1 : TODO

    c_2 : TODO

    nystrom_version : TODO    
    
    seed : int >= 0
        The seed for generating the random matrix W.
    
    References
    ----------
    [X] TODO
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    n = A.shape[0]

    if not isinstance(t, np.ndarray):
        t = np.array(t).reshape(-1)

    N_v_1 = round(n_v * c1)
    N_v_2 = round(n_v * c2)
    N_v_3 = n_v - N_v_1 - N_v_2

    # Polynomial expansion
    g = lambda x: gaussian_kernel(x, n=n, sigma=sigma)
    mu = chebyshev_coefficients(t, m, function=g)

    # Initializations
    W_1 = np.random.rand(n, N_v_1)
    W_2 = np.random.rand(n, N_v_2)
    W_3 = np.random.rand(n, N_v_3)

    Z_1 = chebyshev_recurrence(mu, A, T_0=W_1, final_shape=(n, N_v_1))
    Z_2 = chebyshev_recurrence(mu, A, T_0=W_2, final_shape=(n, N_v_2))
    Z_3 = chebyshev_recurrence(mu, A, T_0=W_3, final_shape=(n, N_v_3))

    phi_tilde = np.zeros(t.shape[0])
    for i in range(t.shape[0]):
        if nystrom_version == "pinv":
            P = _generalized_nystrom_pinv(Z_1[i], Z_2[i], W_2)
        if nystrom_version == "qr":
            P = _generalized_nystrom_qr(Z_1[i], Z_2[i], W_2)
        if nystrom_version == "stable_qr":
            P = _generalized_nystrom_stable_qr(Z_1[i], Z_2[i], W_2)
        phi_tilde[i] = np.trace(P) + (np.trace(W_3.T @ Z_3[i]) + np.trace(W_3.T @ P @ W_3)) / N_v_3

    return phi_tilde


def _hutchinson(A, n_v, seed=0):
    """
    Hutchinson trace estimator.

    Parameters
    ----------
    A : np.ndarray (n, n)
        The matrix for which the trace will be computed.
    n_v : int
        The number of random vectors to be used in total.
    seed : int or None
        The seed for the random number generator.

    Returns
    -------
    t_A : float
        Estimate of trace by the Hutchinson method.
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    n = A.shape[0]

    W = 2.0 * (np.random.rand(n, n_v) > 0.5) - 1.0

    t_A = np.trace(W.T @ A @ W) / n_v
    return t_A


def _hutchpp(A, n_v, sketch_fraction=2/3, seed=0):
    """
    Hutch++ trace estimator.

    Parameters
    ----------
    A : np.ndarray (n, n)
        The matrix for which the trace will be computed.
    n_v : int
        The number of random vectors to be used in total.
    sketch_fraction : float
        The fraction of random vectors which are used for sketching A.
    seed : int or None
        The seed for the random number generator.

    Returns
    -------
    t_A : float
        Estimate of trace by the Hutch++ algorithm.

    References
    ----------
    [6] Meyer et. al. Hutch++: Optimal Stochastic Trace Estimation.
        Proc SIAM Symp Simplicity Algorithms. (2021) 142-155. 
        DOI: 10.1137/1.9781611976496.16
    """   
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix and amount of random mat-vec sketches
    n = A.shape[0]
    n_sketch = round(n_v * sketch_fraction / 2)
    n_hutch  = n_v - 2 * n_sketch

    # Generate sketching matrices
    S = 2.0 * (np.random.rand(n, n_sketch) > 0.5) - 1.0
    G = 2.0 * (np.random.rand(n, n_hutch) > 0.5) - 1.0

    # Generate orthonormal basis of sketch AS
    Q, _ = np.linalg.qr(A @ S)

    # Compute Hutch++ estimate
    G -= Q @ Q.T @ G
    t_A = np.trace(Q.T @ A @ Q) + np.trace(G.T @ A @ G) / n_hutch
    return t_A


def _nahutchpp(A, n_v, c1=1/4, c2=1/2, seed=0):
    """
    Non-adaptive Hutch++ trace estimator.

    Parameters
    ----------
    A : np.ndarray (n, n)
        The matrix for which the trace will be computed.
    n_v : int
        The number of random vectors to be used in total.
    c1 : float
        Fraction of random vectors used for sketching A's QR decomposition.
    c2 : float
        Fraction of random vectors used for sketching A.
    seed : int or None
        The seed for the random number generator.

    Returns
    -------
    t_A : float
        Estimate of trace by the non-adaptive Hutch++ algorithm.

    References
    ----------
    [6] Meyer et. al. Hutch++: Optimal Stochastic Trace Estimation.
        Proc SIAM Symp Simplicity Algorithms. (2021) 142-155. 
        DOI: 10.1137/1.9781611976496.16
    """ 
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix and amount of random mat-vec sketches
    n = A.shape[0]
    n_R_sketch = round(n_v * c1)
    n_S_sketch = round(n_v * c2)
    n_hutch = n_v - n_R_sketch - n_S_sketch

    R = 2.0 * (np.random.rand(n, n_R_sketch) > 0.5) - 1.0
    S = 2.0 * (np.random.rand(n, n_S_sketch) > 0.5) - 1.0
    G = 2.0 * (np.random.rand(n, n_hutch) > 0.5) - 1.0

    Z = A @ R
    W = A @ S

    P = np.linalg.pinv(S.T @ Z)
    trace = np.trace(P @ (W.T @ Z)) + (np.trace(G.T @ A @ G) - np.trace((G.T @ Z) @ P @ (W.T @ G))) / n_hutch
    return trace


def _HDGC(A, t, m, sigma, n_v, estimator=_hutchpp, seed=0):
    """
    Hutchinson-type estimators for Delta-Gauss-Chebyshev method.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Symmetric matrix with eigenvalues between (-1, 1).
    t : np.ndarray (n_t,)
        A set of points at which the DOS is to be evaluated.
    m : int > 0
        Degree of the polynomial.
    sigma : int or float > 0
        Smearing parameter.
    n_v : int > 0
        Number of random vectors in W.
    estimator : function
        The trace estimator.
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.
    """
    # Determine size of matrix i.e. number of eigenvalues 
    n = A.shape[0]

    # Polynomial expansion
    g = lambda x: gaussian_kernel(x, n=n, sigma=sigma)
    mu = chebyshev_coefficients(t, m, function=g)

    # Initializations
    Z = np.zeros((m+1, n, n))
    V_c = np.eye(n)
    V_m = np.zeros((n, n))
    V_p = np.zeros((n, n))

    # Chebyshev recursion
    for l in range(m + 1):
        Z[l] = V_c
        V_p = (1 if l == 0 else 2) * A @ V_c - V_m
        V_m = V_c.copy()
        V_c = V_p.copy()

    # Computation of the approximate spectral density
    g_M = np.tensordot(mu, Z, axes=([1], [0]))
    phi_tilde = np.empty(t.shape[0])
    for i in range(t.shape[0]):
        phi_tilde[i] = estimator(g_M[i], n_v, seed=seed)
    return phi_tilde
