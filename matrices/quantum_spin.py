import numpy as np
import scipy as sp

def hamiltonian(N=20, s=0.5, h=0.3, J=1):

    M = int(2*s+1)

    Jz_T = h*np.ones(N)
    J_T = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if np.abs(i-j)==1: # horizontal neighbor
                J_T[i,j] = J

    N = len(Jz_T)
    s = s
    Jx = J_T
    Jy = J_T
    Jz = Jz_T

    M = int(2*s+1)
    Sx = np.zeros((M,M), "complex")
    Sy = np.zeros((M,M), "complex")
    Sz = np.zeros((M,M), "complex")
    for i in range(M):
        for j in range(M):
            Sx[i,j] = ((i==j+1)+(i+1==j))*np.sqrt(s*(s+1)-(s-i)*(s-j))/2
            Sy[i,j] = ((i+1==j)-(i==j+1))*np.sqrt(s*(s+1)-(s-i)*(s-j))/2j
            Sz[i,j] = (i==j)*(s-i)

    out = sp.sparse.coo_matrix((M**N,M**N))

    for j in range(N):
        if  np.count_nonzero(Jx[:,j]) != 0 or np.count_nonzero(Jy[:,j]) != 0:
            I1 = sp.sparse.eye(M**j)
            I2 = sp.sparse.eye(M**(N-j-1))
            Sxj = sp.sparse.kron(sp.sparse.kron(I1,Sx),I2)
            Syj = sp.sparse.kron(sp.sparse.kron(I1,Sy),I2)
            Szj = sp.sparse.kron(sp.sparse.kron(I1,Sz),I2)

            for i in range(j):
                if Jx[i,j] != 0 or Jy[i,j] != 0:
                    I1 = sp.sparse.eye(M**i)
                    I2 = sp.sparse.eye(M**(N-i-1))
                    Sxi_Sxj = sp.sparse.kron(sp.sparse.kron(I1,Sx),I2)@Sxj
                    Syi_Syj = sp.sparse.kron(sp.sparse.kron(I1,Sy),I2)@Syj
                    #Szi_Szj = sp.sparse.kron(sp.sparse.kron(I1,Sz),I2)@Szj

                    out += (2-(i==j))*Jx[i,j] * Sxi_Sxj
                    out += (2-(i==j))*Jy[i,j] * Syi_Syj

            out += Jz[j] * Szj

    return out.real

def partition_function(beta, N, h, J, E_min):
    
    beta = np.asarray(beta, dtype=np.float64)

    k = np.arange(1, N+1) * np.pi / (N + 1)
    lambda_k = h - 2 * J * np.cos(k)
    exponent = np.sum(np.logaddexp(0, - np.multiply.outer(lambda_k, beta)), axis=0)
    Z_true = np.exp(exponent + beta * (E_min + N * h / 2)) 
    return Z_true