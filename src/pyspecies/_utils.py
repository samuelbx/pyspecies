import numpy as np


def XtoUV(X: np.ndarray):
    K = len(X) // 2
    return X[:K], X[K:]


def UVtoX(U: np.ndarray, V: np.ndarray):
    K = len(U)
    zeroes = np.zeros(2 * K)
    zeroes[:K] = U
    zeroes[K:] = V
    return zeroes


def MergeDiagonals(P: list, Q: list, R: list, S: list):
    return [
        Q[0],
        Q[1],
        Q[2],
        np.concatenate((P[0], Q[3], S[0])),
        np.concatenate((P[1], Q[4], S[1])),
        np.concatenate((P[2], S[2])),
        np.concatenate((P[3], R[0], S[3])),
        np.concatenate((P[4], R[1], S[4])),
        R[2],
        R[3],
        R[4],
    ]


def f(i: int, A: np.ndarray, B: np.ndarray, D2: np.ndarray, R2: np.ndarray):
    A2, AB = A * A, A * B
    center = (
        ((1 - R2[i, 0] + 2 * D2[i, 0]) * A)
        + ((R2[i, i + 1] + 2 * D2[i, i + 1]) * A2)
        + ((R2[i, 2 - i] + 2 * D2[i, 2 - i]) * AB)
    )
    border = D2[i, 0] * A + D2[i, i + 1] * A2 + D2[i, 2 - i] * AB
    return center - np.roll(border, 1) - np.roll(border, -1)


def mu(i: int, A: np.ndarray, B: np.ndarray, D2: np.ndarray):
    return D2[i, 0] * np.ones(len(A)) + 2 * D2[i, i + 1] * A + D2[i, 2 - i] * B


def nu(i: int, A: np.ndarray, B: np.ndarray, D2: np.ndarray, R2: np.ndarray):
    return (
        (1 - R2[i, 0] + 2 * D2[i, 0]) * np.ones(len(A))
        + 2 * (R2[i, i + 1] + 2 * D2[i, i + 1]) * A
        + (R2[i, 2 - i] + 2 * D2[i, 2 - i]) * B
    )


def J2(i: int, A: np.ndarray, D2: np.ndarray, R2: np.ndarray):
    DA = D2[i, 2 - i] * A
    return [
        [-DA[-1]],
        -DA[1:],
        R2[i, 2 - i] * A + 2 * DA,
        -DA[:-1],
        [-DA[0]],
    ]
