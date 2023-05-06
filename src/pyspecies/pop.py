import warnings
from math import ceil
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from pyspecies._euler import back_euler
from pyspecies._utils import UVtoX, XtoUV
from pyspecies.models import SKT


class Pop:
    """A class representing two populations and their interaction model.

    Parameters:
        u0 (Callable): initial concentration of species #1
        v0 (Callable): initial concentration of species #2
        model (SKT): species interaction model
        space (tuple[int, int, int], optional): tuple containing (min_x, max_x, no_points). Defaults to (0, 1, 200).
    """

    def __init__(
        self,
        u0: Callable,
        v0: Callable,
        model: SKT,
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
        self.D, self.R = model.D, model.R
        self.Space = np.linspace(space[0], space[1], K)

        # Compute concentration vectors of initial population
        X0 = UVtoX(u0(self.Space), v0(self.Space))
        if not (X0 >= 0).all():
            raise ValueError("Initial conditions must be positive")

        # Initialize simulation results
        self.Xlist = [X0]
        self.Tlist = np.array([0])

    def sim(self, duration: float, N: int = 100) -> None:
        """Move the simulation forward by a given duration and precision.

        Args:
            duration (float): duration of the simulation in seconds
            N (int, optional): number of time steps (defaults to 100)
        """
        if duration <= 0:
            raise ValueError("Duration must be positive")
        if N < 1:
            raise ValueError("N must be greater than one")

        # Generate corresponding time steps
        Time = np.linspace(0, duration, N)

        # Continue the simulation and save the results
        X0 = self.Xlist[-1].copy()
        self.Xlist += back_euler(X0, Time, self.Space, self.D, self.R)
        self.Tlist = np.append(self.Tlist, self.Tlist[-1] + Time)

    def _formatPlot(self):
        """Helper function to prepare the concentration vs. space graph."""
        # Chart format
        plt.style.use("seaborn-talk")
        fig, ax = plt.subplots()
        ax.set_xlabel("Space")
        ax.set_ylabel("Concentrations")

        # Define the size of the graph axis
        xmin, xmax = np.min(self.Space), np.max(self.Space)
        padx = (xmax - xmin) * 0.05
        ymin, ymax = np.min(self.Xlist), np.max(self.Xlist)
        pady = (ymax - ymin) * 0.05
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

    def anim(self, length: float = 7) -> None:
        """Shows an elegant Matplotlib animation of the steps simulated so far.

        Args:
            length (int, optional): In theory, the duration of the animation in seconds. In practice, it will be relatively longer because of the time needed to render the different images. Defaults to 7.
        """
        if length <= 0:
            raise ValueError("Length must be positive")

        # Prepare the plot
        fig, ax, time_text = self._formatPlot()
        tmax = np.max(self.Tlist)

        # Function called to update the frames
        def _anim(i):
            # Find the closest time step
            j = ceil(i * (len(self.Xlist) - 1) / (length * 50 - 1))

            # Update the graph
            U, V = XtoUV(self.Xlist[j])
            Uarea = ax.fill_between(
                self.Space, U, color="#f44336", alpha=0.5, label="First species"
            )
            Varea = ax.fill_between(
                self.Space, V, color="#3f51b5", alpha=0.5, label="Second species"
            )

            # Update the time text
            display_time = np.round(self.Tlist[j], decimals=2)
            display_percentage = np.round(100 * self.Tlist[j] / tmax, decimals=1)
            time_text.set_text(
                f"Simulation at t={display_time}s ({display_percentage}%)"
            )

            return Uarea, Varea, time_text

        # Animate the graph
        ani = FuncAnimation(
            fig, _anim, frames=range(length * 50), interval=20, blit=True, repeat=True
        )
        plt.legend()
        plt.show()

    def snapshot(self, theta: float, show_prod: False) -> None:
        """Shows a snapshot of the state of the simulation at a certain percentage of completion.

        Args:
            theta (float): simulation completion level, between 0 (initial) and 1 (final)
            show_prod (bool, optional): whether to show the product of the two species. Defaults to False.
        """
        if not (0 <= theta and theta <= 1):
            raise ValueError("theta must lie between 0 and 1")

        # Prepare the plot
        _, ax, time_text = self._formatPlot()
        tmax = np.max(self.Tlist)

        # Find the closest time step
        j = 0
        while self.Tlist[j] < theta * tmax and j < len(self.Tlist) - 2:
            j += 1

        # Update the graph
        U, V = XtoUV(self.Xlist[min(j, len(self.Xlist) - 1)])
        ax.fill_between(
            self.Space, U, color="#f44336", alpha=0.5, label="First species"
        )
        ax.fill_between(
            self.Space, V, color="#3f51b5", alpha=0.5, label="Second species"
        )
        if show_prod:
            ax.plot(self.Space, U * V, color="red")

        # Update text for simulation time
        display_time = np.round(self.Tlist[j], decimals=2)
        time_text.set_text(f"Simulation at t={display_time}s ({theta * 100:.0f}%)")
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
