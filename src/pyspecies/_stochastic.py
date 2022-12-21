from math import ceil

import numpy as np
from tqdm import tqdm


def batch_size(X: tuple[np.ndarray, np.ndarray], tol: float, dx: float) -> int:
    """_summary_

    Args:
        X (tuple[np.ndarray, np.ndarray]): _description_
        tol (float): _description_
        dx (float): _description_

    Returns:
        int: _description_
    """
    assert 0 <= tol < 1
    total_pop = (X[0].sum() + X[1].sum()) * dx
    return max(1, ceil(total_pop * tol))


def compute_steps(
    X0: tuple[np.ndarray, np.ndarray],
    duration: float,
    D: np.ndarray,
    R: np.ndarray,
    tol: float,
    dx: float,
) -> tuple[list, list]:
    """_summary_

    Args:
        X0 (tuple[np.ndarray, np.ndarray]): _description_
        duration (float): _description_
        D (np.ndarray): _description_
        R (np.ndarray): _description_
        tol (float): _description_
        dx (float): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        tuple[list, list]: _description_
    """
    U0, V0 = X0[0].copy(), X0[1].copy()
    # liste des 3-uplets (espèce considérée (0 ou 1), orig, dest) ou 1 unité d'espèce se déplace de orig vers dest
    steps = []
    Tsteps = []  # liste des temps correspondant à chaque saut
    # le nombre de binomial/choice calculées et mémoïsées s'obtient avec la fonction batch_size((U, V), tol, dx)
    # s'arrêter quand Tsteps[-1] (temps de la dernière étape calculée) dépasse duration
    # essayer de faire un TQDM personnalisé en fonction du temps du dernier saut calculé sur duration
    # TODO : implémenter les termes de diffusion quadratique et de réaction
    # en attendant : erreur
    raise NotImplementedError
    return steps, Tsteps


def linearize_time(
    X0: tuple[np.ndarray, np.ndarray],
    steps: list[tuple[int, int, int]],
    Tsteps: list[float],
    Tlist: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """_summary_

    Args:
        X0 (tuple[np.ndarray, np.ndarray]): _description_
        steps (list[tuple[int, int, int]]): _description_
        Tsteps (list[float]): _description_
        Tlist (np.ndarray): _description_

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: _description_
    """
    U0, V0 = X0
    Utemp, Vtemp = U0.copy(), V0.copy()
    Xlist_s = [X0]

    k = 1
    for i in tqdm(range(len(steps))):
        species, orig, dest = steps[i]
        if species == 0:
            Utemp[dest] += 1
            Utemp[orig] += -1
        elif species == 1:
            Vtemp[dest] += 1
            Vtemp[orig] += -1

        if Tsteps[i] >= Tlist[k]:
            Xlist_s.append((Utemp.copy(), Vtemp.copy()))
            k += 1

        if k >= len(steps):
            break

    return Xlist_s
