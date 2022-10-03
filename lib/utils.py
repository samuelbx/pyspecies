import numpy as np


def XtoUV(X):
    return X[::2], X[1::2]


def UVtoX(U, V):
    zeroes = np.zeros(2 * len(U))
    zeroes[::2] = U
    zeroes[1::2] = V
    return zeroes