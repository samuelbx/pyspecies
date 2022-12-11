from typing import Callable

import numpy as np

from pyspecies._anim import Animate
from pyspecies._euler import BackwardEuler
from pyspecies._utils import UVtoX
from pyspecies.models import SKT


class Pop:
    """A class to represent two populations and their interaction model.

    Attributes:
        u0 (function): first species' initial concentration
        v0 (function): second species' initial concentration
        model (SKT): interaction model between the species
        space (tuple, optional): tuple containing (min_x, max_x, no_points).
            Defaults to (0, 1, 200).
    """

    def __init__(
        self, u0: Callable, v0: Callable, model: SKT, space: tuple = (0, 1, 200)
    ):
        assert len(space) == 3, "space must contain 3 elements: min_x, max_x, no_points"

        self.D, self.R = model.D, model.R
        self.Space = np.linspace(space[0], space[1], space[2])
        X0 = UVtoX(u0(self.Space), v0(self.Space))
        assert (X0 >= 0).all(), "Initial conditions must be positive"
        self.Xlist = [X0]
        self.Tlist = np.array([0])

    def sim(self, duration: float, N: int = 100):
        """Move the simulation forward by a given duration and precision.

        Args:
            duration (float): Duration of the simulation
            N (int, optional): Number of time steps. Defaults to 100.
        """
        assert duration > 0, "Duration must be positive"
        assert N >= 1, "N must be greater than one"
        Time = np.linspace(0, duration, N)
        self.Tlist = np.append(self.Tlist, self.Tlist[-1] + Time)
        X0 = self.Xlist[-1].copy()
        self.Xlist = self.Xlist + BackwardEuler(X0, Time, self.Space, self.D, self.R)

    def resetAnim(self):
        """Only keeps the last calculated time step so that
        the next animation starts from it."""
        self.Xlist = [self.Xlist[0]]
        self.Tlist = [self.Tlist[0]]

    def anim(self, length: float = 7):
        """Shows a nice Matplotlib animation of the steps simulated so far.

        Args:
            length (int, optional): In theory, the duration of the animation in seconds.
                In practice, it will be relatively longer due to the display time of
                the different images. Defaults to 7.
        """
        assert len(self.Tlist > 1), "Nothing to animate yet"
        assert length > 0, "Length must be positive"

        K, N = len(self.Space), len(self.Tlist)
        txt = "D={}, R={}, K={}, N={}".format(
            str(self.D).replace("\n", ""), str(self.R).replace("\n", ""), K, N - 2
        )
        Animate(self.Space, self.Xlist, self.Tlist, length=length, text=txt)
