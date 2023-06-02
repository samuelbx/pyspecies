from abc import abstractmethod

import numpy as np
import scipy.sparse as sps

from pyspecies._euler import back_euler
from pyspecies._utils import XtoUV, block_diags, f, merge_diags, mu, nu
from pyspecies.models import Model


class Continuous(Model):
    """Abstract class for models that are solved using backward Euler methods."""

    @abstractmethod
    def funcjac(self, dx: float, dt: float):
        pass

    def sim(self, X0: np.ndarray, Space: np.ndarray, Time: np.ndarray):
        dx, dt = Space[1] - Space[0], Time[1] - Time[0]
        return back_euler(X0, Time, self.funcjac(dx, dt)), Time


class SKT(Continuous):
    """Shigesada-Kawasaki-Teramoto population interaction model.
    See https://doi.org/10.1016/0022-5193(79)90258-3 for more details.

    Attributes:
        D (np.ndarray): diffusion matrix.
        R (np.ndarray): reaction matrix.
    """

    def __init__(self, D: np.ndarray, R: np.ndarray) -> None:
        self.D, self.R = D, R

    def funcjac(self, dx: float, dt: float) -> tuple[np.ndarray, sps.dia_matrix]:
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

        def _funcjac(X: np.ndarray, Xm: np.ndarray):
            # Current and previous concentration vectors
            U, V = XtoUV(X)
            Um, Vm = XtoUV(Xm)

            D2, R2 = dt / dx**2 * self.D, dt * self.R  # Renormalize matrices
            K = len(U)  # Number of space points

            # Compute function
            g = np.concatenate((f(0, U, V, D2, R2) - Um, f(1, V, U, D2, R2) - Vm))

            # Compute jacobian blocks
            r, s = mu(0, U, V, D2), mu(1, V, U, D2)
            JP = [-r[-1:], -r[1:], nu(0, U, V, D2, R2), -r[:-1], -r[:1]]
            JQ, JR = block_diags(0, U, D2, R2), block_diags(1, V, D2, R2)
            JS = [-s[-1:], -s[1:], nu(1, V, U, D2, R2), -s[:-1], -s[:1]]

            # Assemble jacobian matrix in sparse format
            jac = sps.diags(
                merge_diags(JP, JQ, JR, JS),
                [2 * K - 1, K + 1, K, K - 1, 1, 0, -1, -K + 1, -K, -K - 1, -2 * K + 1],
                format="csr",
            )

            return g, jac

        return _funcjac


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
