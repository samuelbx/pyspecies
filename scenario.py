from lib.population import Population
from lib.utils import plateau
import numpy as np


def MixedScenario():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: .2 + plateau(x + .1),
                     v0=lambda x: .4 - plateau(x - .1),
                     D=np.array([[1e-3, 3e-1], [3e-1, 1e-3]]))
    pop.sim(duration=1, N=250)
    pop.sim(duration=10, N=250)
    pop.sim(duration=100, N=250)
    pop.sim(duration=1000, N=250)
    pop.anim()  # pop.animate(filename='mixed') to save as mp4


def Test():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: .2 + plateau(x + .1),
                     v0=lambda x: .4 - plateau(x - .1),
                     D=np.array([[0, 0, 0], [0, 0, 0]]),
                     R=np.array([[1, 1, 1], [0, 0, 0]]))
    pop.sim(duration=10, N=1000)
    pop.anim()


def LinearScenario():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: .2 + plateau(x + .1),
                     v0=lambda x: .4 - plateau(x - .1),
                     D=np.array([[1e-2, 0, 0], [1e-2, 0, 0]]))
    pop.sim(duration=10, N=1000)
    pop.anim()


def SineScenario1():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: np.sin(4 * x + .12)**2,
                     v0=lambda x: np.sin(4 * x - .12)**2,
                     D=np.array([[0, 3e-1], [3e-1, 0]]))
    pop.sim(duration=.1, N=500)
    pop.sim(duration=.9, N=500)
    pop.anim()


def SineScenario2():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: np.cos(2 * x)**2,
                     v0=lambda x: np.sin(6 * x)**2,
                     D=np.array([[0, 2e-2], [2e-2, 0]]))
    pop.sim(duration=1, N=334)
    pop.sim(duration=10, N=333)
    pop.sim(duration=50, N=333)
    pop.anim()


def IntruderScenario():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: 3*plateau(5*x),
                     v0=lambda x: plateau(x/2),
                     D=np.array([[0, 2e-2], [2e-2, 0]]))
    pop.sim(duration=0.1, N=50)
    pop.sim(duration=1, N=50)
    pop.sim(duration=10, N=50)
    pop.sim(duration=100, N=50)
    pop.sim(duration=1000, N=50)
    pop.sim(duration=10000, N=50)
    pop.anim()

Test()

""" Test the function Merge Diagonals:
from lib.euler import MergeDiagonals

K = 5
J1 = [10, [11]*(K-1), [12]*(K), [13]*(K-1), 14]
J2 = [20, [21]*(K-1), [22]*(K), [23]*(K-1), 24]
J3 = [30, [31]*(K-1), [32]*(K), [33]*(K-1), 34]
J4 = [40, [41]*(K-1), [42]*(K), [43]*(K-1), 44]

import scipy.sparse as sp
print(sp.diags(MergeDiagonals(J1,J2,J3,J4), [2 * K - 1, K + 1, K, K - 1, 1, 0, -1, -K + 1, -K, -K - 1, -2 * K + 1]).toarray())
"""