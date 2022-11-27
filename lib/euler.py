import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from tqdm import tqdm

from lib.utils import XtoUV


def MergeDiagonals(P, Q, R, S):
    return [
        [Q[0]],
        Q[1],
        Q[2],
        np.concatenate(([P[0]], Q[3], [S[0]])),
        np.concatenate((P[1], [Q[4]], S[1])),
        np.concatenate((P[2], S[2])),
        np.concatenate((P[3], [R[0]], S[3])),
        np.concatenate(([P[4]], R[1], [S[4]])),
        R[2],
        R[3],
        [R[4]],
    ]


def funcAndJac(
    X: np.ndarray, Xm: np.ndarray, D: np.ndarray, R: np.ndarray, dx: float, dt: float
):
    U, V = XtoUV(X)
    Um, Vm = XtoUV(Xm.copy())
    D2, R2 = dt / dx**2 * D, dt * R
    K = len(U)

    # Compute function
    def f(i, A, B):
        A2, AB = A * A, A * B
        center = (
            ((1 - R2[i, 0] + 2 * D2[i, 0]) * A)
            + ((R2[i, i + 1] + 2 * D2[i, i + 1]) * A2)
            + ((R2[i, 2 - i] + 2 * D2[i, 2 - i]) * AB)
        )
        border = D2[i, 0] * A + D2[i, i + 1] * A2 + D2[i, 2 - i] * AB
        return center - np.roll(border, 1) - np.roll(border, -1)

    g = np.concatenate((f(0, U, V) - Um, f(1, V, U) - Vm))

    # Compute jacobian
    def mu(i, A, B):
        return D2[i, 0] * np.ones(K) + 2 * D2[i, i + 1] * A + D2[i, 2 - i] * B

    def nu(i, A, B):
        return (
            (1 - R2[i, 0] + 2 * D2[i, 0]) * np.ones(K)
            + 2 * (R2[i, i + 1] + 2 * D2[i, i + 1]) * A
            + (R2[i, 2 - i] + 2 * D2[i, 2 - i]) * B
        )

    def J2(i, A):
        return [
            -D2[i, 2 - i] * A[-1],
            -D2[i, 2 - i] * A[1:],
            (R2[i, 2 - i] + 2 * D2[i, 2 - i]) * A,
            -D2[i, 2 - i] * A[:-1],
            -D2[i, 2 - i] * A[0],
        ]

    r, s = mu(0, U, V), mu(1, V, U)

    J1 = [-r[-1], -r[1:], nu(0, U, V), -r[:-1], r[0]]
    J4 = [-s[-1], -s[1:], nu(1, V, U), -s[:-1], -s[0]]

    Jg = sp.diags(
        MergeDiagonals(J1, J2(0, U), J2(1, V), J4),
        [2 * K - 1, K + 1, K, K - 1, 1, 0, -1, -K + 1, -K, -K - 1, -2 * K + 1],
    )

    return g, Jg


def BackwardEuler(
    X0: np.ndarray,
    Time: np.ndarray,
    Space: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    newtThreshold=1e-8,
    max_iter=1000,
):
    X_list = [X0]
    dt = Time[1] - Time[0]
    dx = Space[1] - Space[0]

    for n in tqdm(range(1, len(Time)), "Simulation in progress"):
        Xm = X_list[-1].copy()
        Xk = Xm.copy()

        # Multivariate Newton-Raphson method with sparse jacobian
        for _ in range(max_iter):
            b, A = funcAndJac(Xk, Xm, D, R, dx, dt)
            deltaX = spl.spsolve(A, -b)
            Xk += deltaX

            # Convergence criterion
            if np.linalg.norm(deltaX) < newtThreshold:
                break

        X_list.append(Xk)

    del X_list[0]

    return X_list
