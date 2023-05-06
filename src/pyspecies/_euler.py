import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from pyspecies._utils import XtoUV, block_diags, f, merge_diags, mu, nu


def _funcjac(
    X: np.ndarray, Xm: np.ndarray, D: np.ndarray, R: np.ndarray, dx: float, dt: float
) -> tuple[np.ndarray, sp.dia_matrix]:
    """Computes the function and the sparse Jacobian used in the iteration of Newton's method.

    Args:
        X (np.ndarray): current concentration vector.
        Xm (np.ndarray): previous concentration vector.
        D (np.ndarray): diffusion matrix.
        R (np.ndarray): reaction matrix.
        dx (float): space step.
        dt (float): time step.

    Returns:
        tuple[np.ndarray, sp.dia_matrix]: function and Jacobian.
    """

    # Current and previous concentration vectors
    U, V = XtoUV(X)
    Um, Vm = XtoUV(Xm)

    D2, R2 = dt / dx**2 * D, dt * R  # Renormalize matrices
    K = len(U)  # Number of space points

    # Compute function
    g = np.concatenate((f(0, U, V, D2, R2) - Um, f(1, V, U, D2, R2) - Vm))

    # Compute jacobian blocks
    r, s = mu(0, U, V, D2), mu(1, V, U, D2)
    JP = [-r[-1:], -r[1:], nu(0, U, V, D2, R2), -r[:-1], -r[:1]]
    JQ, JR = block_diags(0, U, D2, R2), block_diags(1, V, D2, R2)
    JS = [-s[-1:], -s[1:], nu(1, V, U, D2, R2), -s[:-1], -s[:1]]

    # Assemble jacobian matrix in sparse format
    jac = sp.diags(
        merge_diags(JP, JQ, JR, JS),
        [2 * K - 1, K + 1, K, K - 1, 1, 0, -1, -K + 1, -K, -K - 1, -2 * K + 1],
        format="csr",
    )

    return g, jac


def _cuthill_mckee(K: int) -> np.ndarray:
    """Cuthill-McKee algorithm for bandwidth reduction.

    Args:
        K (int): number of space points.

    Returns:
        np.ndarray: reverse Cuthill-McKee permutation.
    """

    # Assemble typical Jacobian matrix
    a = np.ones(K)
    d = [a[:1], a[1:], a, a[1:], a[:1]]
    M = sp.diags(
        merge_diags(d, d, d, d),
        [2 * K - 1, K + 1, K, K - 1, 1, 0, -1, -K + 1, -K, -K - 1, -2 * K + 1],
        format="csr",
    )

    # Compute reverse Cuthill-McKee permutation
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
    """Solves the reaction-diffusion system using the backward Euler method.

    Args:
        X0 (np.ndarray): initial concentration vector.
        Time (np.ndarray): time steps.
        Space (np.ndarray): space steps.
        D (np.ndarray): diffusion matrix.
        R (np.ndarray): reaction matrix.
        newt_thres (float, optional): convergence threshold for Newton's method (defaults to 1e-4)
        max_iter (int, optional): maximum number of iterations for Newton's method (defaults to 10)

    Returns:
        list[np.ndarray]: approximate solution at each time step.
    """
    X_list = [X0]
    dx = Space[1] - Space[0]

    # Compute reverse Cuthill-McKee permutation
    perm = _cuthill_mckee(len(X0) // 2)

    for n in tqdm(range(1, len(Time)), "Deterministic simulation"):
        Xm = X_list[-1].copy()
        Xk = Xm.copy()
        dt = Time[n] - Time[n - 1]

        # Newton's method with sparse Jacobian
        for i in range(max_iter):
            # Compute function and Jacobian
            b, A = _funcjac(Xk, Xm, D, R, dx, dt)

            # Solve linear system after applying permutation
            deltaX = spsolve(A[perm, :][:, perm], -b[perm])[np.argsort(perm)]
            Xk += deltaX

            # Stop if convergence is reached
            if np.linalg.norm(deltaX) < newt_thres * np.linalg.norm(Xk):
                break

            # Raise error if maximum number of iterations is reached
            if i == max_iter - 1:
                raise ValueError(
                    f"Newton's method cannot converge in less than {max_iter} steps: aborting."
                )

        X_list.append(Xk)

    # Remove initial condition as it is the final condition of the previous simulation
    del X_list[0]

    return X_list
