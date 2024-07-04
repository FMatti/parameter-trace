import __context__

import numpy as np
from algorithms.krylov_aware import lanczos, block_lanczos


def test_lanczos():
    np.random.seed(0)
    n = 100
    k = 10
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A = A.conj().T + A

    for m in [1, 2, 3, 10, 20]:
        X = np.random.randn(n, m)
        U, T = lanczos(A, X, k, return_matrix=True, extend_matrix=True)

        for i in range(m):
            np.testing.assert_allclose(A @ U[i, :, :k] - U[i] @ T[i], 0, atol=1e-10)

        U, a, b = lanczos(A, X, k, return_matrix=False, extend_matrix=True)
        for i in range(m):
            T = np.diag(b[i, 1:-1], 1) + np.diag(a[i]) + np.diag(b[i, 1:-1], -1)
            T = np.vstack((T, np.eye(1, T.shape[0])[:, ::-1] * b[i, -1]))
            np.testing.assert_allclose(A @ U[i, :, :k] - U[i] @ T, 0, atol=1e-10)


def test_block_lanczos():
    np.random.seed(0)
    n = 100
    k = 10
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A = A.conj().T + A
    
    for m in [1, 2, 3, 10]:
        X = np.random.randn(n, m)
        U, T = block_lanczos(A, X, k, return_matrix=True, extend_matrix=True)
        np.testing.assert_allclose(A @ U[:, :-m] - U @ T, 0, atol=1e-10)
        
        U, a, b = block_lanczos(A, X, k, return_matrix=False, extend_matrix=True)
        T = np.zeros(((k + 1)*m, k*m), dtype=A.dtype)
        for i in range(k):
            T[i*m:(i+1)*m,i*m:(i+1)*m] = a[i]
            T[(i+1)*m:(i+2)*m, i*m:(i+1)*m] = b[i + 1]
            if i < k - 1:
                T[i*m:(i+1)*m, (i+1)*m:(i+2)*m] = b[i + 1].conj().T
        np.testing.assert_allclose(A @ U[:, :-m] - U @ T, 0, atol=1e-10)

def test_krylov_aware():
    return None