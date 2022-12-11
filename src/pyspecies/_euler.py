import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from pyspecies._utils import XtoUV, block_diags, f, merge_diags, mu, nu


def func_and_jac(
    X: np.ndarray, Xm: np.ndarray, D: np.ndarray, R: np.ndarray, dx: float, dt: float
) -> tuple[np.ndarray, sp.dia_matrix]:
    """Computes the function and the Jacobian used when iterating Newton's method.

    Args:
        X (np.ndarray): Concentrations at current time.
        Xm (np.ndarray): Concentrations at previous time.
        D (np.ndarray): Diffusion matrix.
        R (np.ndarray): Reaction matrix.
        dx (float): Discrete space step.
        dt (float): Discrete time step.

    Returns:
        tuple[np.ndarray, sp.dia_matrix]: Relevant function and jacobian.
    """
    U, V = XtoUV(X)
    Um, Vm = XtoUV(Xm)
    D2, R2 = dt / dx**2 * D, dt * R
    K = len(U)

    # Compute function
    g = np.concatenate((f(0, U, V, D2, R2) - Um, f(1, V, U, D2, R2) - Vm))

    # Compute jacobian blocks
    r, s = mu(0, U, V, D2), mu(1, V, U, D2)
    P = [[-r[-1]], -r[1:], nu(0, U, V, D2, R2), -r[:-1], [-r[0]]]
    Q, R = block_diags(0, U, D2, R2), block_diags(1, V, D2, R2)
    S = [[-s[-1]], -s[1:], nu(1, V, U, D2, R2), -s[:-1], [-s[0]]]

    # Build jacobian in diagonal sparse representation
    jac = sp.diags(
        merge_diags(P, Q, R, S),
        [2 * K - 1, K + 1, K, K - 1, 1, 0, -1, -K + 1, -K, -K - 1, -2 * K + 1],
        format="csr",
    )

    return g, jac


def cuthill_permutation(K: int) -> np.ndarray:
    """Compute reverse Cuthill-McKee permutation to lower jacobian bandwidth.

    Args:
        K (int): Space discretization size.

    Returns:
        np.ndarray: Array of permuted row and column indices.
    """
    d = [[1], [1] * (K - 1), [1] * (K), [1] * (K - 1), [1]]
    M = sp.diags(
        merge_diags(d, d, d, d),
        [2 * K - 1, K + 1, K, K - 1, 1, 0, -1, -K + 1, -K, -K - 1, -2 * K + 1],
        format="csr",
    )
    return sp.csgraph.reverse_cuthill_mckee(M, symmetric_mode=True)


def back_euler(
    X0: np.ndarray,
    Time: np.ndarray,
    Space: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    newt_thres: float = 1e-4,
    max_iter: int = 10,
) -> list[np.ndarray]:
    """Runs the Backward-Euler method to solve the SKT model equations.

    Args:
        X0 (np.ndarray): Initial concentrations.
        Time (np.ndarray): Discretized time.
        Space (np.ndarray): Discretized space.
        D (np.ndarray): Diffusion matrix.
        R (np.ndarray): Reaction matrix.
        newt_thres (float, optional): Convergence threshold of Newton's method. Defaults to 1e-4.
        max_iter (int, optional): Maximum number of iterations for Newton's method. Defaults to 10.

    Returns:
        list[np.ndarray]: List of concentration values across time.
    """
    X_list = [X0]
    dx = Space[1] - Space[0]

    # Compute reverse Cuthill-McKee permutation to lower jacobian bandwidth
    perm = cuthill_permutation(len(X0) // 2)

    for n in tqdm(range(1, len(Time)), "Simulation in progress"):
        Xm = X_list[-1].copy()
        Xk = Xm.copy()
        dt = Time[n] - Time[n - 1]

        # Multivariate Newton-Raphson method with sparse jacobian
        for _ in range(max_iter):
            b, A = func_and_jac(Xk, Xm, D, R, dx, dt)
            deltaX = spsolve(A[perm, :][:, perm], -b[perm])[np.argsort(perm)]
            Xk += deltaX

            # Convergence criterion
            if np.linalg.norm(deltaX) < newt_thres:
                break

            # Raise an error if convergence seems impossible
            if _ == max_iter - 1:
                raise ValueError(
                    "Newton's method cannot converge in less than {} steps: aborting.".format(
                        str(max_iter)
                    )
                )

        X_list.append(Xk)

    del X_list[0]

    return X_list
