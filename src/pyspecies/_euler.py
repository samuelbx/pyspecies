import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from pyspecies._utils import J2, MergeDiagonals, XtoUV, f, mu, nu


def funcAndJac(
    X: np.ndarray, Xm: np.ndarray, D: np.ndarray, R: np.ndarray, dx: float, dt: float
):
    U, V = XtoUV(X)
    Um, Vm = XtoUV(Xm)
    D2, R2 = dt / dx**2 * D, dt * R
    K = len(U)

    # Compute function
    g = np.concatenate((f(0, U, V, D2, R2) - Um, f(1, V, U, D2, R2) - Vm))

    # Compute jacobian
    r, s = mu(0, U, V, D2), mu(1, V, U, D2)
    J1 = [-r[-1], -r[1:], nu(0, U, V, D2, R2), -r[:-1], -r[0]]
    J4 = [-s[-1], -s[1:], nu(1, V, U, D2, R2), -s[:-1], -s[0]]
    Jg = sp.diags(
        MergeDiagonals(J1, J2(0, U, D2, R2), J2(1, V, D2, R2), J4),
        [2 * K - 1, K + 1, K, K - 1, 1, 0, -1, -K + 1, -K, -K - 1, -2 * K + 1],
        format="csr",
    )

    return g, Jg


def CuthillPermutation(K):
    J1 = [10, [11] * (K - 1), [12] * (K), [13] * (K - 1), 14]
    J2 = [20, [21] * (K - 1), [22] * (K), [23] * (K - 1), 24]
    J3 = [30, [31] * (K - 1), [32] * (K), [33] * (K - 1), 34]
    J4 = [40, [41] * (K - 1), [42] * (K), [43] * (K - 1), 44]
    M = sp.diags(
        MergeDiagonals(J1, J2, J3, J4),
        [2 * K - 1, K + 1, K, K - 1, 1, 0, -1, -K + 1, -K, -K - 1, -2 * K + 1],
        format="csr",
    )
    return sp.csgraph.reverse_cuthill_mckee(M, symmetric_mode=True)


def BackwardEuler(
    X0: np.ndarray,
    Time: np.ndarray,
    Space: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    newtThreshold=1e-4,
    max_iter=1000,
):
    X_list = [X0]
    perm = CuthillPermutation(len(X0) // 2)
    dx = Space[1] - Space[0]

    for n in tqdm(range(1, len(Time)), "Simulation in progress"):
        Xm = X_list[-1].copy()
        Xk = Xm.copy()
        dt = Time[n] - Time[n - 1]

        # Multivariate Newton-Raphson method with sparse jacobian
        for _ in range(max_iter):
            b, A = funcAndJac(Xk, Xm, D, R, dx, dt)
            deltaX = spsolve(A[perm, :][:, perm], -b[perm])[np.argsort(perm)]
            Xk += deltaX

            if np.linalg.norm(deltaX) < newtThreshold:
                break

        X_list.append(Xk)

    del X_list[0]

    return X_list
