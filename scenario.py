from lib.population import Population
from lib.utils import plateau
import numpy as np


def MixedScenario():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: .2 + plateau(x + .1),
                     v0=lambda x: .4 - plateau(x - .1),
                     D=np.array([[1e-3, 3e-1], [3e-1, 1e-3]]))
    pop.sim(duration=10, N=1000)
    pop.sim(duration=100, N=500)
    pop.sim(duration=300, N=500)
    pop.anim()  # pop.animate(filename='mixed') to save as mp4


def LinearScenario():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: .2 + plateau(x + .1),
                     v0=lambda x: .4 - plateau(x - .1),
                     D=np.array([[1e-2, 0], [0, 1e-2]]))
    pop.sim(duration=10, N=1000)
    pop.anim()


def SineScenario1():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: np.sin(4 * x)**2,
                     v0=lambda x: np.sin(4 * x + .25)**2,
                     D=np.array([[0, 3e-1], [3e-1, 0]]))
    pop.sim(duration=.2, N=500)
    pop.sim(duration=1.3, N=500)
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


# Feel free to choose one of the previous scenarios
SineScenario1()