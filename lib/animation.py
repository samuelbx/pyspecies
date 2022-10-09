from lib.utils import XtoUV
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def Animate(Space, Xlist, length=7, filename='', text=''):
    fig = plt.figure()
    Uline, = plt.plot([], [], color='r', label='Species #1')
    Vline, = plt.plot([], [], color='b', label='Species #2')

    plt.xlabel('Space')
    plt.ylabel('Concentrations')
    plt.legend()

    # Compute plot axis size
    xmin, xmax = np.min(Space), np.max(Space)
    padx = (xmax - xmin) * 0.2
    ymin, ymax = np.min(Xlist), np.max(Xlist)
    pady = (ymax - ymin) * 0.2
    plt.xlim(xmin - padx, xmax + padx)
    plt.ylim(ymin - pady, ymax + pady)

    def anim(i):
        U, V = XtoUV(Xlist[i])
        Uline.set_data(Space, U)
        Vline.set_data(Space, V)
        return Uline, Vline

    Xlist = Xlist[::max(len(Xlist) // (length * 100), 1)]

    ani = animation.FuncAnimation(fig,
                                  anim,
                                  frames=len(Xlist),
                                  interval=max((length * 1000) // len(Xlist), 1),
                                  blit=True,
                                  repeat=True)

    if text:
        plt.title(text)

    plt.show()

    if filename:
        w = animation.writers['ffmpeg']
        w = animation.FFMpegWriter(fps=60, bitrate=1800)
        ani.save(filename + '.mp4', writer=w, dpi=120)
