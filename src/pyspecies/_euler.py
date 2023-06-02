import numpy as np
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def back_euler(
    X0: np.ndarray,
    Time: np.ndarray,
    funcjac: callable,
    newt_thres: float = 1e-4,
    max_iter: int = 10,
) -> list[np.ndarray]:
    """Backward Euler method with Newton-Raphson algorithm.

    Args:
        X0 (np.ndarray): initial concentration vector.
        Time (np.ndarray): time steps.
        funcjac (callable): function and its Jacobian.
        newt_thres (float, optional): convergence threshold for Newton's method (defaults to 1e-4)
        max_iter (int, optional): maximum number of iterations for Newton's method (defaults to 10)

    Returns:
        list[np.ndarray]: approximate solution at each time step.
    """
    X_list = [X0]

    for _ in tqdm(range(1, len(Time)), "Simulation"):
        Xm = X_list[-1].copy()
        Xk = Xm.copy()

        # Newton's method with sparse Jacobian
        for i in range(max_iter):
            # Compute function and Jacobian
            b, A = funcjac(Xk, Xm)

            # Solve linear system after applying eventual permutation
            dX = spsolve(A, -b)
            Xk += dX

            # Stop if convergence is reached
            if np.linalg.norm(dX) < newt_thres * np.linalg.norm(Xk):
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
