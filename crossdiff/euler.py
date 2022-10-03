from crossdiff.utils import XtoUV, UVtoX
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from tqdm import tqdm


# Input : L = (x1,...,xn)
# Output : (x1, 0, x2, 0, ..., 0, xn)
# with optional additional zeroes at the beggining / end
def padWithZeros(L, atStart: int, atEnd: int):
    L2 = np.zeros(2 * len(L) + atStart + atEnd - 1)
    L2[atStart::2] = L
    return L2


# TODO: Verify correctness of matrix
MODES = {
    'P': {'diags': [-2, 0, 2], 'pad': [[0, 1], [0, 1], [0, 1]]},
    'Q': {'diags': [-1, 1, 3], 'pad': [[1, 1], [0, 0], [0, 0]]},
    'R': {'diags': [-3, -1, 1], 'pad': [[0, 0], [0, 0], [1, 1]]},
    'S': {'diags': [-2, 0, 2], 'pad': [[1, 0], [1, 0], [1, 0]]}
}


def M(c1: float, c2: float, W: np.ndarray, der=False, mode=None):
    baseline = 2 * (c1 * np.ones(len(W)) + c2 * W)

    # Set diag, underdiag and overdiag coefficients
    under = -baseline[:-1] / 2
    over = -baseline[:-1] / 2
    center = baseline

    if not der:
        center += np.ones(len(W))

    # Boundary conditions
    coef = int(not der)
    center[0] = coef  # A[0, 0]
    over[0] = -coef  # A[0, 1]
    under[-1] = coef  # A[-1, -2]
    center[-1] = -coef  # A[-1, -1]

    if not mode:
        return sp.diags([under, center, over], [-1, 0, 1])

    # If asked, artificially insert zeroes between coefficients
    # to prepare the final sparse matrix
    else:
        diags = MODES[mode]['diags']
        pad = MODES[mode]['pad']
        under = padWithZeros(under, pad[0][0], pad[0][1])
        center = padWithZeros(center, pad[1][0], pad[1][1])
        over = padWithZeros(over, pad[2][0], pad[2][1])
        return sp.diags([under, center, over], diags)


def g(X: np.ndarray, Xm: np.ndarray, D: np.ndarray, dx: float, dt: float):
    U, V = XtoUV(X)
    Um, Vm = XtoUV(Xm.copy())
    Um[0], Um[-1], Vm[0], Vm[-1] = 0, 0, 0, 0
    Alpha = D * dt / (dx**2)
    return UVtoX(
        M(Alpha[0, 0], Alpha[0, 1], V).dot(U) - Um,
        M(Alpha[1, 1], Alpha[1, 0], U).dot(V) - Vm)


def Jg(X: np.ndarray, D: np.ndarray, dx: float, dt: float):
    U, V = XtoUV(X)
    Alpha = D * dt / (dx**2)
    P = M(Alpha[0, 0], Alpha[0, 1], V, der=False, mode='P')
    Q = M(0, Alpha[0, 1], U, der=True, mode='Q')
    R = M(0, Alpha[1, 0], V, der=True, mode='R')
    S = M(Alpha[1, 1], Alpha[1, 0], U, der=False, mode='S')
    return P + Q + R + S


def BackwardEuler(X0: np.ndarray,
                  Time: np.ndarray,
                  Space: np.ndarray,
                  D: np.ndarray,
                  newtThreshold=1e-8,
                  dvgThreshold=1e-1,
                  max_iter=1000):
    X_list = [X0]
    dt = Time[1] - Time[0]
    dx = Space[1] - Space[0]

    print('dt/(dx^2) =', dt / (dx**2))
    for n in tqdm(range(1, len(Time))):
        Xm = X_list[-1].copy()
        Xk = Xm.copy()

        # Multivariate Newton-Raphson method with sparse jacobian
        for _ in range(max_iter):
            Jac = Jg(Xk, D, dx, dt)
            B = -g(Xk, Xm, D, dx, dt)
            deltaX = spl.spsolve(Jac, B)
            Xk += deltaX

            # Convergence criterion of the Newton method
            norm = np.linalg.norm(deltaX)
            if norm < newtThreshold:
                break

            # Stop the simulation if the artifacts are too important
            if norm > dvgThreshold:
                return X_list

        X_list.append(Xk)

    return X_list