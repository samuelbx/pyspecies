import numpy as np


class SKT:
    """Sheguesada Kawazaki Teramoto population interaction model.

    Attributes:
        D (np.ndarray): diffusion coefficients' matrix
        R (np.ndarray): reaction coefficients' matrix
    """

    def __init__(self, D: np.ndarray, R: np.ndarray):
        self.D, self.R = D, R


class LV(SKT):
    """Lotka-Volterra population interaction model.

    Attributes:
        alpha (float): prey repoduction rate (>= 0)
        beta (float): prey mortality due to predators (>= 0)
        delta (float): predator reproduction due to eating preys (>= 0)
        gamma (float): predator death rate (>= 0)
    """

    def __init__(self, alpha: float, beta: float, delta: float, gamma: float):
        assert (
            alpha >= 0 and beta >= 0 and delta >= 0 and gamma >= 0
        ), "Coefficients must all be positive"
        self.D = np.array([[0, 0, 0], [0, 0, 0]])
        self.R = np.array([[alpha, 0, beta], [-gamma, -delta, 0]])


class CLV(SKT):
    """Competitive Lotka-Volterra population interaction model.

    See https://en.wikipedia.org/wiki/Competitive_Lotka-Volterra_equations.

    Attributes:
        r1 (float): reproduction rate for (1) (positive)
        r2 (float): reproduction rate for (2) (positive)
        K1 (float): carrying capacity for (1) (positive)
        K2 (float): carrying capacity for (2) (positive)
        s12 (float): competitive effect (1) has on (2) (positive)
        s21 (float): competitive effect (2) has on (1) (positive)
    """

    def __init__(
        self, r1: float, r2: float, K1: float, K2: float, s12: float, s21: float
    ):
        assert (
            r1 >= 0 and r2 >= 0 and s12 >= 0 and s21 >= 0
        ), "Coefficients must all be positive"
        self.D = np.array([[0, 0, 0], [0, 0, 0]])
        self.R = np.array(
            [[r1, -r1 / K1, -r1 * s12 / K1], [r2, -r2 * s21 / K2, -r2 / K2]]
        )
