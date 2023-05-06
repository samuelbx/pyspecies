import numpy as np


class SKT:
    """Sheguesada-Kawazaki-Teramoto population interaction model.

    Attributes:
        D (np.ndarray): diffusion matrix.
        R (np.ndarray): reaction matrix.
    """

    def __init__(self, D: np.ndarray, R: np.ndarray) -> None:
        self.D, self.R = D, R


class LV(SKT):
    """Lotka-Volterra population interaction model. Coefficients must all be positive.

    Attributes:
        a (float): prey repoduction rate.
        b (float): prey mortality due to predators.
        c (float): predator reproduction due to eating preys.
        d (float): predator death rate.
    """

    def __init__(self, a: float, b: float, c: float, d: float) -> None:
        if not (a >= 0 and b >= 0 and c >= 0 and d >= 0):
            raise ValueError("Coefficients must all be positive")
        self.D = np.zeros((2, 3))
        self.R = np.array([[a, 0, b], [-d, -c, 0]])


class CLV(SKT):
    """Competitive Lotka-Volterra population interaction model. Coefficients must all be positive.

    See https://en.wikipedia.org/wiki/Competitive_Lotka-Volterra_equations.

    Attributes:
        r1 (float): reproduction rate for (1).
        r2 (float): reproduction rate for (2).
        K1 (float): carrying capacity for (1).
        K2 (float): carrying capacity for (2).
        s12 (float): competitive effect (1) has on (2).
        s21 (float): competitive effect (2) has on (1).
    """

    def __init__(
        self, r1: float, r2: float, K1: float, K2: float, s12: float, s21: float
    ) -> None:
        if not (r1 >= 0 and r2 >= 0 and s12 >= 0 and s21 >= 0):
            raise ValueError("Coefficients must all be positive")
        if not (K1 > 0 and K2 > 0):
            raise ValueError("Carrying capacities must be > 0")
        self.D = np.zeros((2, 3))
        self.R = np.array(
            [[r1, -r1 / K1, -r1 * s12 / K1], [r2, -r2 * s21 / K2, -r2 / K2]]
        )
