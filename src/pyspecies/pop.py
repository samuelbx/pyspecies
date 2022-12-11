import warnings
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from pyspecies._anim import Animate
from pyspecies._euler import BackwardEuler
from pyspecies._utils import UVtoX, XtoUV
from pyspecies.models import SKT


class Pop:
    """A class representing two populations and their interaction model.

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
        if len(space) != 3:
            raise ValueError("space must contain 3 elements: min_x, max_x, no_points")
        if space[0] > space[1]:
            raise ValueError("max_x must be greater than min_x")
        no_space = space[2]
        if no_space < 10:
            warnings.warn(
                "There should be at least 10 points in space for the program to work. Number of points has automatically been set to 10."
            )
            no_space = 10
        self.D, self.R = model.D, model.R
        self.Space = np.linspace(space[0], space[1], no_space)

        X0 = UVtoX(u0(self.Space), v0(self.Space))
        if not (X0 >= 0).all():
            raise ValueError("Initial conditions must be positive")
        self.Xlist = [X0]
        self.Tlist = np.array([0])

    def sim(self, duration: float, N: int = 100):
        """Move the simulation forward by a given duration and precision.

        Args:
            duration (float): Duration of the simulation
            N (int, optional): Number of time steps. Defaults to 100.
        """
        if duration <= 0:
            raise ValueError("Duration must be positive")
        if N < 1:
            raise ValueError("N must be greater than one")

        Time = np.linspace(0, duration, N)
        self.Tlist = np.append(self.Tlist, self.Tlist[-1] + Time)
        X0 = self.Xlist[-1].copy()
        self.Xlist = self.Xlist + BackwardEuler(X0, Time, self.Space, self.D, self.R)

    def resetAnim(self):
        """Only keeps the last calculated time step so that the next animation starts from it."""
        self.Xlist = [self.Xlist[0]]
        self.Tlist = [self.Tlist[0]]

    def anim(self, length: float = 7):
        """Shows a nice Matplotlib animation of the steps simulated so far.

        Args:
            length (int, optional): In theory, the duration of the animation in seconds.
                In practice, it will be relatively longer due to the display time of
                the different images. Defaults to 7.
        """
        if length <= 0:
            raise ValueError("Length must be positive")

        K, N = len(self.Space), len(self.Tlist)
        txt = "D={}, R={}, K={}, N={}".format(
            str(self.D).replace("\n", ""), str(self.R).replace("\n", ""), K, N - 2
        )
        Animate(self.Space, self.Xlist, self.Tlist, length=length, text=txt)

    def heatmap(self):
        """Shows a nice 2D heatmap of the dominating species over time and space."""
        grid = np.zeros((len(self.Tlist), len(self.Space)))
        for i, X in enumerate(self.Xlist):
            U, V = XtoUV(X)
            grid[i, :] = U - V

        grid = grid[:-1, :-1]
        fig, ax = plt.subplots()
        Ubound, Vbound = max(grid.max(), 0), max(-grid.min(), 0)
        bound = max(Ubound, Vbound)
        c = ax.pcolormesh(
            self.Space, self.Tlist, grid, cmap="RdBu_r", vmin=-bound, vmax=bound
        )
        ax.set_title("Domination heatmap (u: red, v: blue)")
        ax.set_xlabel("Space")
        ax.set_ylabel("Time")
        fig.colorbar(c, ax=ax)
        plt.show()
