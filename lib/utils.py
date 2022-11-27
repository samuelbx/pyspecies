import numpy as np


def XtoUV(X):
    K = len(X) // 2
    return X[:K], X[K:]


def UVtoX(U, V):
    K = len(U)
    zeroes = np.zeros(2 * K)
    zeroes[:K] = U
    zeroes[K:] = V
    return zeroes


# C^inf function which is null on negative numbers
def exp_cinf(x):
    return np.exp(-1 / np.maximum(x, 1e-18)) * np.int64(x > 0)


# Plateau function (support = [-0.5, 0.5])
def plateau(x):
    return (30 * exp_cinf(x + 0.5) * exp_cinf(-x + 0.5)) ** 2
