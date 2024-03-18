"""
Matrices
--------

Assembly routines for the matrices used to test the algorithms.
"""

import functools
import itertools

import numpy as np
import scipy as sp


def second_derivative_finite_difference(N, h=1, bc="dirichlet"):
    """
    Matrix which applies the fininte difference second derivative to a vector.

    Parameters
    ----------
    N : int > 0
        Number of grid points.
    h : int or float > 0
        Spacing between the grid points.
    bc : "dirichlet" or "periodic"
        Nature of the boundary conditions.

    Returns
    -------
    A : np.ndarray of shape (N, N)
        The finite difference matrix corresponding to the problem.
    """
    A = sp.sparse.diags(
        diagonals=np.multiply.outer([1, -2, 1], np.ones(N)),
        offsets=[-1, 0, 1],
        shape=(N, N),
        format="lil",
    )

    if bc == "dirichlet":
        pass
    elif bc == "periodic":
        A[-1, 0] = 1
        A[0, -1] = 1
    else:
        raise ValueError("Boundary condition bc='{}' is unknown.".format(bc))

    return A / h**2


def laplace_finite_difference(N, h=1, dim=3, bc="dirichlet"):
    """
    Matrix which applies the fininte difference second derivative to a vector.

    Parameters
    ----------
    N : int > 0
        Number of grid points.
    h : int or float > 0
        Spacing between the grid points.
    dim : int > 0
        The spatial dimension of the grid.
    bc : str {"dirichlet", "periodic"}
        Nature of the boundary conditions.

    Returns
    -------
    L : np.ndarray of shape (N, N)
        The finite difference matrix corresponding to the Laplace operator.

    References
    ----------
    [3] D. Kressner. Low Rank Approximation Techniques. Lecture Notes. (2022)
        Lecture 7, Slide 24.
    """
    A = -second_derivative_finite_difference(N=N, h=h, bc=bc)

    # Generate Laplace matrix using Kronecker products as seen in [3]
    L = sp.sparse.csr_matrix((N**dim, N**dim))
    for i in range(dim):
        L += functools.reduce(
            sp.sparse.kron,
            i * [sp.sparse.eye(N)] + [A] + (dim - i - 1) * [sp.sparse.eye(N)],
        )

    return L


def regular_grid(a=0, b=1, N=10, dim=3):
    """
    Generate a regularly spaced grid in all dimensions.

    Parameters
    ----------
    a : int or float
        Starting point of grid in all dimensions.
    b : int or float
        Ending point of grid in all dimensions.
    N : int
        Number of grid-points.
    dim : int > 0
        The spatial dimension of the grid.

    Returns
    -------
    grid_points : np.ndarray of shape (n,)
        The Gaussian function evaulated at all points X.
    """
    grid = np.meshgrid(*dim * [np.linspace(a, b, N)])
    grid_points = np.vstack([x.flatten() for x in grid]).T
    return grid_points


def gaussian_well(X, mu=None, var=None, scaling_factor=1.0):
    """
    Gaussian well potential:

        g(X) = scaling_factor * exp( - (X - mu)^T * inv(var) * (X - mu) )

    Parameters
    ----------
    X : int, float, or np.ndarray of shape (n, dim)
        Point(s) where the Gaussian function should be evaluated.
        Format: array([[x1, y1, ...], [x2, y2, ...], ...])
    mu : int, float, or np.ndarray of shape (dim,)
        Multivariate mean vector of the Gaussian function. If None is given, the
        zero vector is taken. If int or float are given, they are extended to
        the constant mean vector of appropriate dimension (given by X).
    var : int, float, or np.ndarray of shape (dim,) or (dim, dim)
        Multivariate variance of the Gaussian function. If None is given, the
        identity matrix is taken. If int or float are given, they are extended
        to the constant variance matrix of appropriate dimension (given by X).
    scaling_factor : int or float
        The scaling factor by which each Gaussian well is scaled.

    Returns
    -------
    g(X) : np.ndarray of shape (n, dim)
        The Gaussian function evaulated at all points X.

    [1] Simon J.D. Prince. Computer Vision: Models, Learning, and Inference.
        Cambridge University Press. (2012)
    """
    # Convert X to a numpy array  of shape (n, d) to make computations easier
    if not isinstance(X, np.ndarray):
        X = np.array(X).reshape(-1, 1)
    if len(X.shape) < 2:
        X = X.reshape(1, -1)  # Need to break tie between (n, 1) and (1, dim) vectors
    dim = X.shape[1]

    # Parse the mean vector
    if mu is None:
        mu = np.zeros(dim)
    elif not isinstance(mu, np.ndarray):
        mu = mu * np.ones(dim)

    # Parse the variance matrix
    if var is None:
        var = np.ones(dim)
    elif not isinstance(var, np.ndarray):
        var = var * np.ones(dim)
    if len(var.shape) < 2:
        var = np.diag(var)

    diff = X - mu
    exponent = -0.5 * np.sum(diff * np.dot(diff, np.linalg.inv(var)), axis=1)
    return scaling_factor * np.exp(exponent)


def periodic_gaussian_well(X, n=1, L=6, var=1.0, scaling_factor=1.0):
    """
    Potential constructed using periodic repetitions of a Gaussian unit cell.

    Parameters
    ----------
    X : int, float, or np.ndarray of shape (n, dim)
        Point at which the potential function should be evaluated.
    n : int
        Number of repeated unit cells in each dimension.
    L : int or float
        Length of the unit cells.
    var : int or float
        Variance of the Gaussians.
    scaling_factor : int or float
        The scaling factor by which each Gaussian well is scaled.

    Returns
    -------
    potential : np.ndarray of shape (n,)
        The periodic Gaussian potential function evaulated at all points X.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017).
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    if np.min(X) < 0 or np.max(X) > n * L:
        raise ValueError("Point x={} is outside of specified domain.".format(X))

    if not isinstance(X, np.ndarray):
        X = np.array(X).reshape(-1, 1)
    if len(X.shape) < 2:
        X = X.reshape(1, -1)
    dim = X.shape[1]

    # Determine distance from which contributions of wells are still considered 
    cut_off = 1e-16
    gaussian_reach = 2 * var / L**2 * np.log(np.abs(scaling_factor) / cut_off)
    k = int(np.ceil(np.sqrt(gaussian_reach) - 0.5))

    # Generate the indices of all cells (e.g. 1st (0, 0, 0), 2nd (0, 0, 1), ...)
    cell_indices = itertools.product(*dim * [range(-k, n + k)])

    # Add up the contributions from all cells
    potential = np.zeros(X.shape[0])
    for cell_index in cell_indices:
        mu = L / 2 * np.ones(dim) + L * np.array(cell_index)
        potential += gaussian_well(X, mu=mu, var=var, scaling_factor=scaling_factor)

    return potential


def ModES3D(n=1, L=6, h=0.6, dim=3, bc="periodic", beta=2.0, alpha=-4.0):
    """
    Generate the example matrices 'ModES3D_X' from [2].

    Parameters
    ----------
    n : int > 0
        Number of unit cells of Gaussians in each dimension (X = n**dim).
    L : int or float > 0
        Length of the unit cells.
    h : int or float > 0
        Spacing between the grid points.
    dim : int > 0
        The spatial dimension of the grid.
    bc : str {"dirichlet", "periodic"}
        Nature of the boundary conditions.
    beta : int or float > 0
        The variance of the Gaussians.
    alpha :  int or float
        Scaling factor of the Gaussians.

    Returns
    -------
    np.ndarray of shape (N, N)
        The matrix corresponding to the operator.

    Remarks
    -------
    The Gaussians are constructed using the following formula:
 
        g(r) = prefactor / √(2π * var) * exp(- r^2 / (2 * var)) + shift

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017).
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    N = n * round(L / h)
    A = laplace_finite_difference(N=N, h=h, dim=dim, bc=bc)
    grid_points = regular_grid(a=0, b=L * n - h, N=N, dim=dim)
    V = sp.sparse.diags(
        periodic_gaussian_well(grid_points, L=L, n=n, var=beta**2, scaling_factor=alpha)
    )

    return A + V


def uniform(n=5000, a=-1, b=1, density=0.0001, seed=0):
    """
    Generate a sparse matrix with uniformly spaced eigenvalues in [a, b].

    Parameters
    ----------
    n : int > 0
        The size of the matrix.
    a : float
        The smallest eigenvalue.
    b : float
        The largest eigenvalue.
    density : float
        The density of the resulting matrix.
    seed : int
        The seed used to generate the random matrix.

    Returns
    -------
    np.ndarray of shape (n, n)
        The matrix corresponding to the operator.
    """
    # Fix the seed
    np.random.seed(seed)

    D = sp.sparse.diags(np.linspace(a, b, n))
    U = sp.sparse.random(n, n, density=density)
    Q, _ = sp.linalg.qr(U.toarray())
    Q = sp.sparse.csr_matrix(Q)

    return Q @ D @ Q.T


def gaussian_orthogonal_ensemble(n=5000, seed=0):
    # Fix the seed
    np.random.seed(seed)
    
    A = np.random.randn(n, n)
    return (A + A.T) / np.sqrt(2)


# --- Unused implementations ---


def _laplace_finite_difference_eigvals(n=1, L=6, h=0.6, dim=3):
    """
    Eigenvalues of periodic finite difference discretization of the Laplacian.

    Parameters
    ----------
    n : int > 0
        Number of unit cells of Gaussians in each dimension (X = n**dim).
    L : int or float > 0
        Length of the unit cells.
    h : int or float > 0
        Spacing between the grid points.
    dim : int > 0
        The spatial dimension of the grid.

    Returns
    -------
    eigvals : np.ndarray of shape (N,)
        The eigenvalues corresponding to the problem.

    References
    ----------
    [4] Wikipedia. Eigenvalues and eigenvectors of the second derivative. 2023.
        https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors_of_the_second_derivative
    """
    n = L / h
    j = np.arange(n) + np.arange(n) % 2
    eigvals1d = 4 / h**2 * np.sin(np.pi * j / (2 * n)) ** 2
    eigvals = functools.reduce(np.add.outer, dim * [eigvals1d]).flatten()
    return eigvals
