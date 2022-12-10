from math import ceil

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from pyspecies.utils import XtoUV


def Animate(Space, Xlist, Tlist, length=7, text=""):
    plt.style.use("seaborn-talk")

    fig, ax = plt.subplots()

    ax.set_xlabel("Space")
    ax.set_ylabel("Concentrations")

    # Compute plot axis size
    xmin, xmax = np.min(Space), np.max(Space)
    padx = (xmax - xmin) * .05
    ymin, ymax = np.min(Xlist), np.max(Xlist)
    pady = (ymax - ymin) * .05

    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + 2 * pady)

    subt = ax.text(
        .5,
        .95,
        "",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    if text:
        ax.set_title(text)

    def anim(i):
        j = ceil(i * (len(Xlist) - 1) / (length * 50 - 1))

        U, V = XtoUV(Xlist[j])
        Uarea = ax.fill_between(Space, U, color="#f44336", alpha=0.5)
        Varea = ax.fill_between(Space, V, color="#3f51b5", alpha=0.5)
        subt.set_text(
            "Population dynamics simulation at t={}s".format(
                str(np.round(Tlist[j], decimals=2))
            )
        )

        return Uarea, Varea, subt

    ani = animation.FuncAnimation(
        fig, anim, frames=range(length * 50), interval=20, blit=True, repeat=True
    )

    plt.show()
