import numpy as np

from pyspecies.anim import Animate
from pyspecies.euler import BackwardEuler
from pyspecies.models import SKT
from pyspecies.utils import UVtoX


class Pop:
    def __init__(self, space: tuple, u0, v0, model: SKT):
        self.D, self.R = model.D, model.R
        self.Space = np.linspace(space[0], space[1], space[2])
        X0 = UVtoX(u0(self.Space), v0(self.Space))
        assert (X0 >= 0).all()
        self.Xlist = [X0]
        self.Tlist = np.array([0])

    def sim(self, duration: float, N=100):
        Time = np.linspace(0, duration, N)
        self.Tlist = np.append(self.Tlist, self.Tlist[-1] + Time)
        X0 = self.Xlist[-1].copy()
        self.Xlist = self.Xlist + BackwardEuler(X0, Time, self.Space, self.D, self.R)

    def resetAnim(self):
        self.Xlist = [self.Xlist[0]]
        self.Tlist = [self.Tlist[0]]

    def anim(self, length=7):
        assert len(self.Tlist > 1), "Nothing to animate yet"

        K, N = len(self.Space), len(self.Tlist)
        txt = "D={}, R={}, K={}, N={}".format(
            str(self.D).replace("\n", ""), str(self.R).replace("\n", ""), K, N - 2
        )
        return Animate(self.Space, self.Xlist, self.Tlist, length=length, text=txt)
