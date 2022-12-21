import warnings
from math import ceil
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad

from pyspecies._euler import back_euler
from pyspecies._stochastic import compute_steps, linearize_time
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
        self,
        u0: Callable,
        v0: Callable,
        model: SKT,
        space: tuple = (0, 1, 200),
        with_stochastic: bool = False,
        no_bins: int = 15,
        tolerance: float = 0.05,
    ) -> None:
        if not (0 <= tolerance and tolerance <= 0.1):
            raise ValueError(
                "Tolerance should be set between 0 (no approximation) and 0.1 (important approximation)."
            )
        if len(space) != 3:
            raise ValueError("space must contain 3 elements: min_x, max_x, no_points.")
        if space[0] > space[1]:
            raise ValueError("max_x must be greater than min_x.")
        no_space = space[2]
        if no_space < 10:
            warnings.warn(
                "There should be at least 10 points in space for the program to work. Number of points has automatically been set to 10."
            )
            no_space = 10
        # Store environment data
        self.D, self.R = model.D, model.R
        self.Space = np.linspace(space[0], space[1], no_space)

        # Compute initial population concentration vectors
        X0 = UVtoX(u0(self.Space), v0(self.Space))
        if not (X0 >= 0).all():
            raise ValueError("Initial conditions must be positive")
        self.Xlist = [X0]
        self.Tlist = np.array([0])

        # Stochastic simulation parameters
        self.with_stochastic = with_stochastic
        self.no_bins = no_bins
        self.tolerance = tolerance

        # Compute initial conditions for stochastic simulation if required
        if self.with_stochastic:
            space_s = np.linspace(space[0], space[1], no_bins + 1)
            dx = space_s[1] - space_s[0]
            U0, V0 = np.zeros(no_bins), np.zeros(no_bins)
            for i in range(no_bins):
                U0[i] = quad(u0, space_s[i], space_s[i + 1])[0] / dx
                V0[i] = quad(u0, space_s[i], space_s[i + 1])[0] / dx
            X0_s = (U0, V0)
            self.Xlist_s = [X0_s]
            self.Space_s = space_s[:-1] + dx / 2
            plt.bar(self.Space_s, U0, dx)
            plt.plot(self.Space, u0(self.Space))
            plt.show()

    def sim(self, duration: float, N: int = 100) -> None:
        """Move the simulation forward by a given duration and precision.

        Args:
            duration (float): Duration of the simulation
            N (int, optional): Number of time steps. Defaults to 100.
        """
        if duration <= 0:
            raise ValueError("Duration must be positive")
        if N < 1:
            raise ValueError("N must be greater than one")

        # Update time array
        Time = np.linspace(0, duration, N)

        # Continue the simulation and store its results
        X0 = self.Xlist[-1].copy()
        self.Xlist += back_euler(X0, Time, self.Space, self.D, self.R)
        self.Tlist = np.append(self.Tlist, self.Tlist[-1] + Time)

        # Stochastic simulation
        if self.with_stochastic:
            X0_s = self.Xlist_s[-1]
            dx = self.Space_s[1] - self.Space_s[0]
            steps, Tsteps = compute_steps(
                X0_s, duration, self.D, self.R, self.tolerance, dx
            )
            self.Xlist_s += linearize_time(X0_s, steps, Tsteps, self.Tlist)

    def anim(self, length: float = 7) -> None:
        """Shows a nice Matplotlib animation of the steps simulated so far.

        Args:
            length (int, optional): In theory, the duration of the animation in seconds.
                In practice, it will be relatively longer due to the display time of
                the different images. Defaults to 7.
        """
        if length <= 0:
            raise ValueError("Length must be positive")

        # Format the plot
        plt.style.use("seaborn-talk")
        fig, ax = plt.subplots()
        ax.set_xlabel("Space")
        ax.set_ylabel("Concentrations")

        # Define plot axis size
        tmax = np.max(self.Tlist)
        xmin, xmax = np.min(self.Space), np.max(self.Space)
        padx = (xmax - xmin) * 0.05
        ymin, ymax = np.min(self.Xlist), np.max(self.Xlist)
        pady = (ymax - ymin) * 0.05
        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim(ymin - pady, ymax + 2 * pady)

        # Simulation time label on the animation
        time_text = ax.text(
            0.5,
            0.95,
            "",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        # Function called to update frames
        def anim(i):
            # Time corresponding to frame index
            j = ceil(i * (len(self.Xlist) - 1) / (length * 50 - 1))

            # Draw both population's area
            U, V = XtoUV(self.Xlist[j])
            Uarea = ax.fill_between(self.Space, U, color="#f44336", alpha=0.5)
            Varea = ax.fill_between(self.Space, V, color="#3f51b5", alpha=0.5)

            # Update text for simulation time
            time_text.set_text(
                "Simulation at t={}s ({}%)".format(
                    str(np.round(self.Tlist[j], decimals=2)),
                    str(int(100 * self.Tlist[j] / tmax)),
                )
            )

            return Uarea, Varea, time_text

        # Show the animation
        ani = FuncAnimation(
            fig, anim, frames=range(length * 50), interval=20, blit=True, repeat=True
        )
        plt.show()

    def resetAnim(self) -> None:
        """Only keeps the last calculated time step so that the next animation starts from it."""
        self.Xlist = self.Xlist[:-1]
        self.Tlist = self.Tlist[:-1]
        if self.with_stochastic:
            self.Xlist_s = [self.Xlist_s[-1]]

    def heatmap(self) -> None:
        """Shows a nice 2D heatmap of the dominating species over time and space."""
        # Build heatmap grid
        grid = np.zeros((len(self.Tlist), len(self.Space)))
        for i, X in enumerate(self.Xlist):
            U, V = XtoUV(X)
            grid[i, :] = U - V
        grid = grid[:-1, :-1]

        # Show heatmap
        fig, ax = plt.subplots()
        Ubound, Vbound = max(grid.max(), 0), max(-grid.min(), 0)
        bound = max(Ubound, Vbound)  # Color palette boundary
        c = ax.pcolormesh(
            self.Space, self.Tlist, grid, cmap="RdBu_r", vmin=-bound, vmax=bound
        )
        ax.set_title("Domination heatmap (u: red, v: blue)")
        ax.set_xlabel("Space")
        ax.set_ylabel("Time")
        fig.colorbar(c, ax=ax)
        plt.show()
