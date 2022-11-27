import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from lib.utils import XtoUV

t = None


# TODO: fix animation length calculation
def Animate(Space, Xlist, Tlist, length=15, filename="", text=""):
    plt.style.use("seaborn-talk")

    fig, ax = plt.subplots()

    ax.set_xlabel("Space")
    ax.set_ylabel("Concentrations")
    ax.legend()

    # Compute plot axis size
    xmin, xmax = np.min(Space), np.max(Space)
    padx = (xmax - xmin) * 0.05
    ymin, ymax = np.min(Xlist), np.max(Xlist)
    pady = (ymax - ymin) * 0.05

    st = plt.suptitle("", fontweight="bold")

    def anim(i):
        global t
        if i == 0:
            t = time.time()
        elapsed = time.time() - t
        j = int(np.floor(elapsed / length * len(Xlist))) % len(Xlist)
        U, V = XtoUV(Xlist[j])
        ax.clear()
        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim(ymin - pady, ymax + pady)
        ax.fill_between(Space, U, color="#f44336", alpha=0.5)
        ax.fill_between(Space, V, color="#3f51b5", alpha=0.5)

        if text:
            ax.set_title(text)
        st.set_text(
            "Population dynamics simulation at t={}s".format(
                str(np.round(Tlist[j], decimals=2))
            )
        )

    ani = animation.FuncAnimation(
        fig, anim, frames=length * 1000 * 100, interval=1, blit=False, repeat=True
    )

    plt.show()

    if filename:
        w = animation.writers["ffmpeg"]
        w = animation.FFMpegWriter(fps=30, bitrate=1800)
        ani.save(filename + ".mp4", writer=w, dpi=200)
        print("TODO: fix saving")
