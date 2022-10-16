import numpy as np


def XtoUV(X):
    return X[::2], X[1::2]


def UVtoX(U, V):
    zeroes = np.zeros(2 * len(U))
    zeroes[::2] = U
    zeroes[1::2] = V
    return zeroes


# C^inf function which is null on negative numbers
def exp_cinf(x):
    return np.exp(-1 / np.maximum(x, 1e-18)) * np.int64(x > 0)


# Plateau function (support = [-0.5, 0.5])
def plateau(x):
    return (30 * exp_cinf(x + .5) * exp_cinf(-x + .5))**2