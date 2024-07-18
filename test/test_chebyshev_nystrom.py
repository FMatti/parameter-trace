import __context__

import numpy as np
from algorithms.chebyshev_nystrom import chebyshev_expansion, square_chebyshev_expansion
from algorithms.helpers import gaussian_kernel


def test_chebyshev_expansion():
    np.random.seed(0)

    # Fixed parameters
    m_list = [100, 200, 500, 1000]
    sigma = 0.01

    # For some linearly spaced t in [-1, 1], check if approximation is exact
    t_list = np.linspace(-1, 1, 100)
    s_list = np.linspace(-1, 1, 100)
    kernel = lambda t, x: gaussian_kernel(t, x, sigma=sigma)

    # Test standard Chebyshev expansion of kernel against theoretical bound
    bound = lambda sigma, m: np.sqrt(2 * np.e / np.pi) / sigma ** 2 * (1 + sigma) ** (-m)

    for m in m_list:
        # Compute function values from Chebyshev approximation
        mu = chebyshev_expansion(kernel, t_list, m)
        for i, t in enumerate(t_list):
            truth = gaussian_kernel(t, s_list, sigma=sigma)
            approx = np.polynomial.chebyshev.Chebyshev(mu[i])(s_list)

            np.testing.assert_allclose(approx, truth, atol=bound(sigma, m))

    # Test non-negative Chebyshev expansion of kernel against theoretical bound
    nonnegative_bound = lambda sigma, m:  2 * np.sqrt(2) * (1 + sigma * np.sqrt(np.pi) * bound(np.sqrt(2)*sigma, m/2)) * bound(np.sqrt(2)*sigma, m/2)

    for m in m_list:
        # Compute function values from Chebyshev approximation
        mu = chebyshev_expansion(kernel, t_list, m, nonnegative=True)
        for i, t in enumerate(t_list):
            truth = gaussian_kernel(t, s_list, sigma=sigma)
            approx = np.polynomial.chebyshev.Chebyshev(mu[i])(s_list)

            np.testing.assert_allclose(approx, truth, atol=nonnegative_bound(sigma, m))

def test_square_chebyshev_expansion():
    np.random.seed(0)

    m_list = [100, 200, 500, 1000]
    for m in m_list:
        mu = np.random.randn(m + 1)
        nu_numpy = (np.polynomial.chebyshev.Chebyshev(mu) ** 2).coef
        nu_fast = square_chebyshev_expansion(mu)

        np.testing.assert_allclose(nu_fast, nu_numpy)

    np.random.seed(0)

    # Fixed parameters
    m_list = [100, 200, 500, 1000]
    sigma = 0.01

    # For some linearly spaced t in [-1, 1], check if approximation is exact
    t_list = np.linspace(-1, 1, 100)
    s_list = np.linspace(-1, 1, 100)

    # Test standard Chebyshev expansion of kernel against theoretical bound
    bound = lambda sigma, m: np.sqrt(2 * np.e / np.pi) / sigma ** 2 * (1 + sigma) ** (-m)
    squared_bound = lambda sigma, m:  (2 / (sigma * np.sqrt(2 * np.pi)) + bound(sigma, m)) * bound(sigma, m)

    for m in m_list:
        kernel = lambda t, x: gaussian_kernel(t, x, sigma=sigma)

        # Compute function values from Chebyshev approximation
        mu = chebyshev_expansion(kernel, t_list, m)
        nu = square_chebyshev_expansion(mu)
        for i, t in enumerate(t_list):
            truth = gaussian_kernel(t, s_list, sigma=sigma) ** 2
            approx = np.polynomial.chebyshev.Chebyshev(nu[i])(s_list)

            np.testing.assert_allclose(approx, truth, atol=squared_bound(sigma, m))
