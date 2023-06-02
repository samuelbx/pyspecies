import warnings
from math import ceil
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from pyspecies._utils import UVtoX, XtoUV
from pyspecies.models import Model


FPS = 50


class Pop:
    """A class representing two populations and their interaction model.

    Parameters:
        u0 (Callable): initial concentration of species #1
        v0 (Callable): initial concentration of species #2
        model (Model): species interaction model
        space (tuple[int, int, int], optional): tuple containing (min_x, max_x, no_points). Defaults to (0, 1, 200).
    """

    def __init__(
        self,
        u0: Callable,
        v0: Callable,
        model: Model,
        space: tuple[int, int, int] = (0, 1, 200),
    ) -> None:
        if len(space) != 3:
            raise ValueError("space must contain 3 elements: min_x, max_x, no_points")
        if space[0] > space[1]:
            raise ValueError("max_x must be greater than min_x")
        K = space[2]
        if K < 10:
            warnings.warn(
                "The space must have at least 10 points for the program to work properly. The number of points has been automatically set to 10."
            )
            K = 10

        # Store diffusion and reaction matrices and discretized space
        self.model = model
        self.Space = np.linspace(space[0], space[1], K)

        # Compute concentration vectors of initial population
        X0 = UVtoX(u0(self.Space), v0(self.Space))
        if not (X0 >= 0).all():
            raise ValueError("Initial conditions must be positive")

        # Initialize simulation results
        self.Xlist = [X0]
        self.Tlist = np.array([0])

    def sim(self, duration: float = -1, N: int = -1) -> None:
        """Move the simulation forward by a given duration and precision.

        Args:
            duration (float): duration of the simulation in seconds
            N (int, optional): number of time steps (defaults to 100)
        """
        if duration == -1 and N == -1:
            raise ValueError("Either duration or N must be specified")
        if duration != -1 and N == -1:
            N = 100

        # Generate corresponding time steps
        Time = np.linspace(0, duration, N)

        # Continue the simulation and save the results
        X0 = self.Xlist[-1].copy()
        Xlist_n, Tlist_n = self.model.sim(X0, self.Space, Time)
        self.Xlist += Xlist_n
        self.Tlist = np.append(self.Tlist, self.Tlist[-1] + Tlist_n)

    def _formatPlot(self):
        """Helper function to prepare the concentration vs. space graph."""
        # Chart format
        plt.style.use("seaborn-talk")
        fig, ax = plt.subplots()
        ax.set_xlabel("Space")
        ax.set_ylabel("Concentrations")

        # Define the size of the graph axis
        xmin, xmax = np.min(self.Space), np.max(self.Space)
        ymin, ymax = np.min(self.Xlist), np.max(self.Xlist)
        padx, pady = (xmax - xmin) * 0.05, (ymax - ymin) * 0.05
        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim(ymin - pady, ymax + 2 * pady)

        # Add a text for the simulation time
        time_text = ax.text(
            0.5,
            0.95,
            "",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        return fig, ax, time_text

    def _plot(self, ax: plt.Axes, time_text, j: int):
        # Update the graph
        U, V = XtoUV(self.Xlist[j])
        Uarea = ax.fill_between(
            self.Space, U, color="#f44336", alpha=0.5, label="First species"
        )
        Varea = ax.fill_between(
            self.Space, V, color="#3f51b5", alpha=0.5, label="Second species"
        )

        # Update the time text
        t = f"{self.Tlist[j]:.3f}"
        p = f"{100 * self.Tlist[j] / np.max(self.Tlist):.1f}"
        time_text.set_text(f"Simulation at t={t}s ({p}%)")

        return Uarea, Varea, time_text

    def anim(self, length: float = 7, filename="") -> None:
        """Shows an elegant Matplotlib animation of the steps simulated so far.

        Args:
            length (int, optional): In theory, the duration of the animation in seconds. In practice, it will be relatively longer because of the time needed to render the different images. Defaults to 7.
        """
        if length <= 0:
            raise ValueError("Length must be positive")

        # Prepare the plot
        fig, ax, time_text = self._formatPlot()

        # Function called to update the frames
        def _anim(i):
            nonlocal time_text
            j = ceil(i * (len(self.Xlist) - 1) / (length * FPS - 1))
            return self._plot(ax, time_text, j)

        # Animate the graph
        ani = FuncAnimation(
            fig,
            _anim,
            frames=range(length * FPS),
            interval=1000 // FPS,
            blit=True,
        )

        # Save the animation
        if filename != "":
            ani.save(filename)

        plt.legend()
        plt.show()

    def snapshot(self, theta: float) -> None:
        """Shows a snapshot of the state of the simulation at a certain percentage of completion.

        Args:
            theta (float): simulation completion level, between 0 (initial) and 1 (final)
            show_prod (bool, optional): whether to show the product of the two species. Defaults to False.
        """
        if not (0 <= theta and theta <= 1):
            raise ValueError("theta must lie between 0 and 1")

        # Find the closest time step
        j, tmax = 0, np.max(self.Tlist)
        while self.Tlist[j] < theta * tmax and j < len(self.Tlist) - 2:
            j += 1

        # Prepare the plot
        _, ax, time_text = self._formatPlot()
        self._plot(ax, time_text, j)
        plt.legend()
        plt.show()

    def heatmap(self) -> None:
        """Displays a nice 2D heatmap of dominant species over time and space."""
        # Prepare the data
        grid = np.zeros((len(self.Tlist), len(self.Space)))
        for i, X in enumerate(self.Xlist):
            U, V = XtoUV(X)
            grid[i, :] = U - V
        grid = grid[:-1, :-1]

        # Show the heatmap
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
